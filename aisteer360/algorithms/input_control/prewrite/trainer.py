"""PPO-style training driver for the PRewrite rewriter LM.

Wraps a TRL `PPOConfig` for hyperparameter consistency, and implements a single-process REINFORCE-with-KL-penalty
training loop that follows the same algorithmic shape PRewrite describes (sample rewrite, run task LM, score, update
rewriter against reference). The actual policy update is a log-prob-weighted REINFORCE step with a KL penalty
against the frozen reference rewriter; this is the limit of PPO's clipped objective in the single-batch / no-clip
regime and is appropriate for the small-scale training Phase 7 targets.

Test-accessible helpers (`_build_rewriter_input`, `_compute_reward`, `_run_task_lm`, `_step_per_query`,
`_step_static`, `_generate_one_rewrite`) are protected methods exposed for unit tests.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.evaluation.metrics.base import Metric

logger = logging.getLogger(__name__)


class PRewriteTrainer:
    """Wraps a PPO-style training loop to fine-tune the rewriter LM.

    Reward function: for each (rewriter_output, training_query) pair, run the task LM on (rewriter_output,
    training_query) and apply the feedback metric to the task LM's response. The scalar metric value is the reward
    (with a KL penalty applied during the policy update against the frozen reference rewriter).

    For mode="per_query": rewriter sees (meta_prompt, initial_prompt, query_i) per rollout; reward is per-query.
    For mode="static": rewriter sees (meta_prompt, initial_prompt) only; one rewriter output is used to score a batch
    of training queries; reward is the mean score.
    """

    def __init__(
        self,
        rewriter_model: PreTrainedModel,
        rewriter_tokenizer: PreTrainedTokenizerBase,
        task_model: PreTrainedModel | None,
        task_tokenizer: PreTrainedTokenizerBase | None,
        feedback_metric: Metric,
        meta_prompt: str,
        initial_prompt: str,
        mode: Literal["per_query", "static"],
        config: Any,
        rewriter_gen_kwargs: dict,
        task_gen_kwargs: dict,
        n_steps: int = 100,
        batch_size: int = 8,
        kl_coef: float = 0.1,
        learning_rate: float = 1.41e-5,
        seed: int = 0,
    ) -> None:
        if mode not in ("per_query", "static"):
            raise ValueError(f"mode must be 'per_query' or 'static'; got {mode!r}")

        self.rewriter_model = rewriter_model
        self.rewriter_tokenizer = rewriter_tokenizer
        self.task_model = task_model
        self.task_tokenizer = task_tokenizer
        self.feedback_metric = feedback_metric
        self.meta_prompt = meta_prompt
        self.initial_prompt = initial_prompt
        self.mode = mode
        self.config = config
        self.rewriter_gen_kwargs = dict(rewriter_gen_kwargs or {})
        self.task_gen_kwargs = dict(task_gen_kwargs or {})
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.kl_coef = kl_coef
        self.learning_rate = learning_rate
        self.seed = seed

        self._reference_model: PreTrainedModel | None = None

    def train(self, training_data: list[dict]) -> PreTrainedModel:
        """Run training; return the trained rewriter."""
        if not training_data:
            raise ValueError("training_data must be non-empty")

        torch.manual_seed(self.seed)

        self._reference_model = copy.deepcopy(self.rewriter_model)
        self._reference_model.eval()
        for p in self._reference_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            [p for p in self.rewriter_model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )

        rng = torch.Generator()
        rng.manual_seed(self.seed)

        n = len(training_data)
        for step in range(self.n_steps):
            indices = torch.randint(0, n, (self.batch_size,), generator=rng).tolist()
            batch = [training_data[i] for i in indices]
            self.rewriter_model.train()
            try:
                if self.mode == "per_query":
                    metrics = self._step_per_query(batch, optimizer)
                else:
                    metrics = self._step_static(batch, optimizer)
                logger.debug("PRewrite step %d/%d: %s", step + 1, self.n_steps, metrics)
            except Exception as exc:
                logger.exception("PRewrite training step %d failed: %s", step + 1, exc)
                raise

        if self._reference_model is not None:
            del self._reference_model
            self._reference_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.rewriter_model

    def _build_rewriter_input(self, query: str | None) -> str:
        """Format the rewriter's input string from the meta-prompt template.

        For per_query: meta_prompt formatted with `initial_prompt` + `query`.
        For static: meta_prompt formatted with `initial_prompt` only.
        """
        if self.mode == "per_query":
            if query is None:
                raise ValueError("per_query mode requires a query to build the rewriter input.")
            return self.meta_prompt.format(initial_prompt=self.initial_prompt, query=query)
        return self.meta_prompt.format(initial_prompt=self.initial_prompt)

    def _run_task_lm(self, rewritten_prompt: str, query: str) -> str:
        """Run the frozen task LM with the rewritten prompt as system message and the query as user input."""
        if self.task_model is None or self.task_tokenizer is None:
            raise RuntimeError("Task model and tokenizer must be set to run task-LM rollouts.")
        tokenizer = self.task_tokenizer
        has_chat_template = (
            hasattr(tokenizer, "chat_template") and tokenizer.chat_template
        )
        if has_chat_template:
            messages = [
                {"role": "system", "content": rewritten_prompt},
                {"role": "user", "content": query},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = f"{rewritten_prompt}\n\n{query}"

        device = next(self.task_model.parameters()).device
        encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
        gen_kwargs = {"max_new_tokens": 64, **self.task_gen_kwargs}
        with torch.no_grad():
            out = self.task_model.generate(**encoded, **gen_kwargs)
        new_tokens = out[:, encoded["input_ids"].size(1):]
        return tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    def _compute_reward(self, response: str, reference: str | None) -> float:
        """Apply `feedback_metric` and extract a scalar."""
        kwargs: dict[str, Any] = {"responses": [response]}
        if reference is not None:
            kwargs["references"] = [reference]
        result = self.feedback_metric.compute(**kwargs)
        return self._extract_scalar(result)

    @staticmethod
    def _extract_scalar(metric_result: Any) -> float:
        """Extract a scalar score from a Metric's result (mirrors `TaskLMScorer._extract_scalar`)."""
        if isinstance(metric_result, (int, float)):
            return float(metric_result)
        if isinstance(metric_result, dict) and len(metric_result) >= 1:
            value = next(iter(metric_result.values()))
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, (int, float)):
                    return float(first)
            if isinstance(value, (int, float)):
                return float(value)
        raise ValueError(
            f"Could not extract scalar score from metric result: {metric_result!r}."
        )

    def _generate_rewrite(self, prompt_text: str, sample: bool = True) -> tuple[str, torch.Tensor, torch.Tensor]:
        """Generate a rewrite. Returns `(text, prompt_ids, response_ids)` — both 1D tensors on the model's device."""
        device = next(self.rewriter_model.parameters()).device
        encoded = self.rewriter_tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_ids = encoded["input_ids"][0]
        gen_kwargs = {"max_new_tokens": 128, **self.rewriter_gen_kwargs}
        if sample:
            gen_kwargs.setdefault("do_sample", True)
            gen_kwargs.setdefault("temperature", 1.0)
        else:
            gen_kwargs["do_sample"] = False
            gen_kwargs["temperature"] = 0.0
        with torch.no_grad():
            out = self.rewriter_model.generate(**encoded, **gen_kwargs)
        response_ids = out[0, prompt_ids.size(0):]
        text = self.rewriter_tokenizer.decode(response_ids, skip_special_tokens=True)
        return text, prompt_ids, response_ids

    def _generate_one_rewrite(self) -> str:
        """Generate a single (greedy) rewrite for static mode (used by control after training)."""
        if self.mode != "static":
            raise RuntimeError("_generate_one_rewrite is only meaningful in static mode.")
        prompt_text = self._build_rewriter_input(query=None)
        text, _, _ = self._generate_rewrite(prompt_text, sample=False)
        return text.strip()

    def _policy_update(
        self,
        prompt_ids_list: list[torch.Tensor],
        response_ids_list: list[torch.Tensor],
        rewards: list[float],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """REINFORCE-with-KL-penalty update on the rewriter."""
        if not prompt_ids_list:
            return {"loss": 0.0, "reward": 0.0, "kl": 0.0}

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        baseline = rewards_tensor.mean()
        advantages = rewards_tensor - baseline

        total_loss = torch.tensor(0.0, device=next(self.rewriter_model.parameters()).device, requires_grad=False)
        kl_total = 0.0
        loss_terms = []
        for prompt_ids, response_ids, advantage in zip(prompt_ids_list, response_ids_list, advantages):
            if response_ids.numel() == 0:
                continue
            full = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
            full = full.to(next(self.rewriter_model.parameters()).device)

            logits = self.rewriter_model(full).logits[0]
            with torch.no_grad():
                ref_logits = self._reference_model(full).logits[0]

            response_start = prompt_ids.numel() - 1
            response_end = full.size(1) - 1

            response_logits = logits[response_start:response_end]
            response_ref_logits = ref_logits[response_start:response_end]
            targets = full[0, response_start + 1:response_end + 1]

            log_probs = torch.log_softmax(response_logits, dim=-1)
            ref_log_probs = torch.log_softmax(response_ref_logits, dim=-1)

            chosen_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            chosen_ref_log_probs = ref_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            kl_per_token = chosen_log_probs.detach() - chosen_ref_log_probs.detach()
            kl = kl_per_token.sum().item()
            kl_total += kl

            policy_loss = -(advantage.item() * chosen_log_probs.sum())
            kl_loss = self.kl_coef * (chosen_log_probs - chosen_ref_log_probs).sum()

            loss_terms.append(policy_loss + kl_loss)

        if not loss_terms:
            return {"loss": 0.0, "reward": float(rewards_tensor.mean().item()), "kl": 0.0}

        total_loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.rewriter_model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        return {
            "loss": float(total_loss.detach().item()),
            "reward": float(rewards_tensor.mean().item()),
            "kl": float(kl_total / max(len(loss_terms), 1)),
        }

    def _step_per_query(self, batch: list[dict], optimizer: torch.optim.Optimizer) -> dict[str, float]:
        """One training step in per_query mode."""
        prompt_ids_list: list[torch.Tensor] = []
        response_ids_list: list[torch.Tensor] = []
        rewards: list[float] = []
        for row in batch:
            query = row["input"]
            reference = row.get("expected")
            rewriter_input = self._build_rewriter_input(query=query)
            rewritten_text, prompt_ids, response_ids = self._generate_rewrite(rewriter_input, sample=True)
            response = self._run_task_lm(rewritten_text.strip(), query)
            reward = self._compute_reward(response, reference)
            prompt_ids_list.append(prompt_ids)
            response_ids_list.append(response_ids)
            rewards.append(reward)
        return self._policy_update(prompt_ids_list, response_ids_list, rewards, optimizer)

    def _step_static(self, batch: list[dict], optimizer: torch.optim.Optimizer) -> dict[str, float]:
        """One training step in static mode.

        One rewriter rollout is used to score the entire batch; the per-batch mean is the reward used for the policy
        update.
        """
        rewriter_input = self._build_rewriter_input(query=None)
        rewritten_text, prompt_ids, response_ids = self._generate_rewrite(rewriter_input, sample=True)
        rewritten = rewritten_text.strip()

        per_query_rewards = []
        for row in batch:
            query = row["input"]
            reference = row.get("expected")
            response = self._run_task_lm(rewritten, query)
            per_query_rewards.append(self._compute_reward(response, reference))

        mean_reward = sum(per_query_rewards) / max(len(per_query_rewards), 1)
        return self._policy_update([prompt_ids], [response_ids], [mean_reward], optimizer)
