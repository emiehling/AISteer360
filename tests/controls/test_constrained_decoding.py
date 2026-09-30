"""Tests for `ConstrainedDecoding`: declarative source validation, requirements per arm, the
in-process xgrammar-compiled automaton, and the automaton-object configuration."""
import re

import pytest
import torch
from transformers import LogitsProcessor

from steerability.algorithms.core.execution import BackendSpec, Capability, ConstraintSource
from steerability.algorithms.core.internals.probes import Probe, ProbeSet
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.common.processors.constraint import ConstraintProcessor
from steerability.algorithms.output_control.constrained_decoding import ConstrainedDecoding
from steerability.algorithms.output_control.constrained_decoding.utils.automaton import compile_constraint_automaton
from steerability.algorithms.output_control.phased_decoding.control import PhasedDecoding
from steerability.algorithms.output_control.routed_decoding import P, Route, RoutedDecoding, Router, generate, respond
from steerability.algorithms.output_control.search_decoding.control import SearchDecoding
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

# a seed under seed_scope="item" decodes the items of a call one at a time, and seed_scope="dispatch" batches them
SEED_SCOPES = ("item", "dispatch")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=2, hidden=16, heads=2)


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


def _constrained_text(tokenizer, row_ids) -> str:
    """Concatenate a row's non-special tokens, i.e., the string the grammar matched."""
    tokens = tokenizer.convert_ids_to_tokens(row_ids.tolist())
    return "".join(token for token in tokens if token not in tokenizer.all_special_tokens)


def _permitted(processor, prefix_ids) -> list[list[int]]:
    """Run `processor` on `prefix_ids` and return the token ids left finite in each row."""
    out = processor(prefix_ids, torch.zeros(prefix_ids.size(0), len(wordlevel_tokenizer())))
    return [torch.isfinite(row).nonzero().flatten().tolist() for row in out]


def _extend(prefix_ids, new_ids) -> torch.Tensor:
    """Append the rows of `new_ids` to the rows of `prefix_ids`."""
    return torch.cat([prefix_ids, torch.tensor(new_ids)], dim=1)


class _ScriptedBias(LogitsProcessor):
    """Add per-row biases chosen from the generated step, the row index, and the row's last token.

    Applied after the constraint, so a bias never re-enables a masked token.
    """

    def __init__(self, bias_fn):
        self.bias_fn = bias_fn
        self.prompt_length: int | None = None

    def __call__(self, input_ids, scores):
        if self.prompt_length is None:
            self.prompt_length = input_ids.size(1)
        step = input_ids.size(1) - self.prompt_length
        scores = scores.clone()
        for row in range(input_ids.size(0)):
            for token_id, value in self.bias_fn(step, row, int(input_ids[row, -1])).items():
                scores[row, token_id] += value
        return scores


class _PromptRowBias(LogitsProcessor):
    """Bias the k-th prompt row seen toward `token_ids[k % len(token_ids)]`.

    Prompt rows are counted over the rows of each call and across calls. The candidates of a prompt
    therefore start the same choices whether they decode in one batch or one at a time. A prompt row is
    a row of the first call's length.
    """

    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids
        self.prompt_length: int | None = None
        self.seen = 0

    def __call__(self, input_ids, scores):
        if self.prompt_length is None:
            self.prompt_length = input_ids.size(1)
        if input_ids.size(1) != self.prompt_length:
            return scores
        scores = scores.clone()
        for row in range(input_ids.size(0)):
            scores[row, self.token_ids[self.seen % len(self.token_ids)]] += 100.0
            self.seen += 1
        return scores


class TestArgs:

    def test_convenience_fields_build_the_source(self):
        control = ConstrainedDecoding(choice=["cat", "dog"])
        assert control.source == ConstraintSource(kind="choice", value=("cat", "dog"))
        control = ConstrainedDecoding(regex="cat|dog")
        assert control.source.kind == "regex"

    def test_exactly_one_constraint_required(self):
        with pytest.raises(ValueError, match="exactly one"):
            ConstrainedDecoding()
        with pytest.raises(ValueError, match="exactly one"):
            ConstrainedDecoding(regex="a", choice=["b"])

    def test_source_mapping_coerces(self):
        control = ConstrainedDecoding(source={"kind": "regex", "value": "cat"})
        assert isinstance(control.source, ConstraintSource)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown constraint kind"):
            ConstraintSource(kind="template", value="x")


class TestRequirements:

    def test_declarative_source_is_portable(self):
        control = ConstrainedDecoding(json_schema='{"type": "object"}', include_in_scoring=False)
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[control])
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        assert report.supported("generate")

    def test_automaton_object_is_in_process_only(self):
        class _NullAutomaton:
            def reset(self, prefix_ids):
                pass

            def allowed(self, prefix_ids):
                return torch.tensor([0])

        control = ConstrainedDecoding(automaton=_NullAutomaton(), include_in_scoring=False)
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[control])
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        (failure,) = report.failures_for("generate")
        assert failure.message == (
            "ConstrainedDecoding is unsupported at generate on backend kind 'vllm': missing "
            "IN_PROCESS_TORCH; a live automaton object has no declarative form; construct the "
            "control with a ConstraintSource (or json_schema/regex/grammar/choice) or run this "
            "pipeline on the huggingface backend."
        )

    def test_scoring_participation_requires_in_process(self):
        control = ConstrainedDecoding(regex="cat", include_in_scoring=True)
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[control])
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        assert report.supported("generate")
        assert not report.supported("score")
        opted_out = SteeringPipeline(
            model_name_or_path="m",
            controls=[ConstrainedDecoding(regex="cat", include_in_scoring=False)],
        ).check(backend=BackendSpec(kind="vllm", model="m"))
        assert opted_out.supported("score")

    def test_stale_engine_range_names_the_kind(self):
        from steerability.algorithms.core.execution import BackendCapabilities, ConstraintKinds, evaluate_support

        control = ConstrainedDecoding(grammar='root ::= "a"', include_in_scoring=False)
        stale = BackendCapabilities(
            atoms=frozenset({Capability.GUIDED_DECODING}),
            constraint_kinds=ConstraintKinds(constraints=frozenset({"json_schema"})),
        )
        spec = BackendSpec(kind="vllm", model="m")
        report = evaluate_support([control], spec, stale)
        (failure,) = report.failures_for("generate")
        assert "ConstraintKinds(grammar)" in failure.message


class TestInProcessArm:

    def test_choice_constraint_masks_generation(self, model, tokenizer):
        control = ConstrainedDecoding(choice=["cat", "dog"], include_in_scoring=False)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()
        text = pipeline.generate(text="the mat sat on the", max_new_tokens=4, do_sample=False)
        assert text.strip() in ("cat", "dog")

    def test_live_automaton_drives_the_processor(self, model, tokenizer):
        forced = tokenizer.convert_tokens_to_ids("mat")

        class _ForcedAutomaton:
            def reset(self, prefix_ids):
                pass

            def allowed(self, prefix_ids):
                return torch.tensor([forced])

        control = ConstrainedDecoding(automaton=_ForcedAutomaton(), include_in_scoring=False)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()
        output = pipeline.generate(
            text="the cat sat", max_new_tokens=3, do_sample=False, return_output=True,
        )
        assert output.output_ids[0].tolist() == [forced] * 3

    def test_export_constraint_returns_the_source(self):
        control = ConstrainedDecoding(regex="cat|dog")
        assert control.export_constraint() == ConstraintSource(kind="regex", value="cat|dog")
        control = ConstrainedDecoding(automaton=object())
        assert control.export_constraint() is None

    def test_choice_constraint_tracks_each_candidate_row(self, model, tokenizer):
        control = ConstrainedDecoding(choice=["catsat", "dogran"], include_in_scoring=False)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()
        first_tokens = tokenizer.convert_tokens_to_ids(["cat", "dog"])
        # the two candidate rows start different choices
        bias = _ScriptedBias(lambda step, row, last: {first_tokens[row]: 100.0} if step == 0 else {})
        output = pipeline.generate(
            text="the mat sat on the", max_new_tokens=4, do_sample=True, num_return_sequences=2,
            eos_token_id=tokenizer.eos_token_id, logits_processor=[bias], return_output=True, seed=0,
        )
        texts = [_constrained_text(tokenizer, row) for row in output.output_ids]
        assert texts == ["catsat", "dogran"]

    def test_regex_constraint_under_beams(self, model, tokenizer):
        pattern = "(catsat|dogran)"
        control = ConstrainedDecoding(regex=pattern, include_in_scoring=False)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()
        cat, dog, ran = tokenizer.convert_tokens_to_ids(["cat", "dog", "ran"])

        # the "dog" beam ranks first, and the "cat" beam strongly prefers a continuation only the
        # "dog" beam's grammar state permits
        def bias_fn(step, row, last):
            if step == 0:
                return {dog: 50.0, cat: 40.0}
            return {ran: 100.0} if last == cat else {}

        output = pipeline.generate(
            text="the mat sat on the", max_new_tokens=4, num_beams=2, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, logits_processor=[_ScriptedBias(bias_fn)], return_output=True,
        )
        assert re.fullmatch(pattern, _constrained_text(tokenizer, output.output_ids[0]))

    def test_rows_decoded_one_at_a_time_each_follow_the_grammar(self, model, tokenizer):
        never = Probe(
            model_type="llama", location="layer_input", pooling="mean",
            layer_ids=[1], weights={1: torch.ones(16)}, bias=-1e9,
        )
        rules = Router(
            routes=[Route("unreached", when=P("never"), action=respond("the mat"))], default_action=generate(),
        )
        router = RoutedDecoding(probes=ProbeSet({"never": never}), rules=rules)
        control = ConstrainedDecoding(choice=["catsat", "dogran"], include_in_scoring=False)
        pipeline = SteeringPipeline(controls=[control, router], model=model, tokenizer=tokenizer)
        pipeline.steer()
        the, cat, mat = tokenizer.convert_tokens_to_ids(["the", "cat", "mat"])
        pad = tokenizer.pad_token_id
        # the second prompt begins with the first one once the pads are removed
        input_ids = torch.tensor([[pad, pad, 0, the, mat], [0, the, mat, the, cat]])
        outputs = pipeline.generate(
            input_ids=input_ids, attention_mask=(input_ids != pad).long(), max_new_tokens=3, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, return_output=True,
        )
        texts = [_constrained_text(tokenizer, output.output_ids[0]) for output in outputs]
        assert all(text in ("catsat", "dogran") for text in texts), texts

    def test_generated_phases_over_prompts_of_different_lengths_follow_the_grammar(self, model, tokenizer):
        pattern = "(cat|dog)*sat(ran)"
        control = ConstrainedDecoding(regex=pattern, include_in_scoring=False)
        phased = PhasedDecoding(plan=[{"generate": {"until": "sat", "budget": 4}}, {"generate": {}}])
        pipeline = SteeringPipeline(controls=[control, phased], model=model, tokenizer=tokenizer)
        pipeline.steer()
        cat, sat, mat, dog, ran = tokenizer.convert_tokens_to_ids(["cat", "sat", "mat", "dog", "ran"])
        # the longer prompt ends its first phase after one token and the shorter one after two
        next_token = {mat: sat, dog: cat, cat: sat, sat: ran}
        bias = _ScriptedBias(lambda step, row, last: {next_token[last]: 100.0} if last in next_token else {})
        outputs = pipeline.generate(
            text=["the cat sat on the mat", "the dog"], max_new_tokens=6, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, logits_processor=[bias], return_output=True,
        )
        for output in outputs:
            assert re.fullmatch(pattern, _constrained_text(tokenizer, output.output_ids[0]))

    def test_candidates_after_a_one_token_phase_follow_the_grammar_under_either_seed_scope(self, model, tokenizer):
        texts = {}
        for seed_scope in SEED_SCOPES:
            control = ConstrainedDecoding(choice=["catsat", "dogran"], include_in_scoring=False)
            phased = PhasedDecoding(plan=[{"generate": {"budget": 1}}, {"generate": {"budget": 3}}])
            pipeline = SteeringPipeline(controls=[control, phased], model=model, tokenizer=tokenizer)
            pipeline.steer()
            # the two candidates start different choices in the one-token phase
            bias = _PromptRowBias(tokenizer.convert_tokens_to_ids(["cat", "dog"]))
            output = pipeline.generate(
                text="the mat sat on the", n=2, max_new_tokens=6, do_sample=True, seed=1, seed_scope=seed_scope,
                eos_token_id=tokenizer.eos_token_id, logits_processor=[bias], return_output=True,
            )
            texts[seed_scope] = [_constrained_text(tokenizer, row) for row in output.output_ids]
        assert texts["item"] == texts["dispatch"] == ["catsat", "dogran"]

    def test_single_token_segments_follow_the_grammar_under_either_seed_scope(self, model, tokenizer):
        # the "dog" segment ranks above the "cat" segment, which makes the "cat" row decode second in the next
        # iteration; the complete "cat" answer ranks above the complete "dog" answer, and any continuation
        # outside the choice ranks first
        ranking = {"dog": 1.0, "cat": 0.0, "dog ran": 2.0, "cat sat": 3.0}
        texts = {}
        for seed_scope in SEED_SCOPES:
            control = ConstrainedDecoding(choice=["catsat", "dogran"], include_in_scoring=False)
            search = SearchDecoding(
                scorer=lambda prompt, continuations, params: [ranking.get(text, 10.0) for text in continuations],
                segment_len=1, num_candidates=2, keep_k=2, max_iterations=4, propose_mode="sample",
            )
            pipeline = SteeringPipeline(controls=[control, search], model=model, tokenizer=tokenizer)
            pipeline.steer()
            # every search proposes one "cat" and one "dog" segment from the prompt
            bias = _PromptRowBias(tokenizer.convert_tokens_to_ids(["cat", "dog"]))
            output = pipeline.generate(
                text="the mat sat on the", n=2, max_new_tokens=4, seed=1, seed_scope=seed_scope,
                eos_token_id=tokenizer.eos_token_id, logits_processor=[bias], return_output=True,
            )
            texts[seed_scope] = [_constrained_text(tokenizer, row) for row in output.output_ids]
        assert texts["item"] == texts["dispatch"] == ["catsat", "catsat"]

    def test_prompt_that_extends_an_earlier_prompt_and_its_answer_follows_the_grammar(self, model, tokenizer):
        the, cat, sat, mat, dog = tokenizer.convert_tokens_to_ids(["the", "cat", "sat", "mat", "dog"])
        pad = tokenizer.pad_token_id
        # the second prompt is the first prompt, the answer it receives, and a further turn
        input_ids = torch.tensor([[pad, pad, pad, pad, 0, the, mat], [0, the, mat, cat, sat, the, dog]])
        texts = {}
        for seed_scope in SEED_SCOPES:
            control = ConstrainedDecoding(choice=["catsat", "dogran"], include_in_scoring=False)
            pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
            pipeline.steer()
            outputs = pipeline.generate(
                input_ids=input_ids, attention_mask=(input_ids != pad).long(), max_new_tokens=4, do_sample=True,
                seed=1, seed_scope=seed_scope, eos_token_id=tokenizer.eos_token_id,
                logits_processor=[_ScriptedBias(lambda step, row, last: {cat: 50.0})], return_output=True,
            )
            texts[seed_scope] = [_constrained_text(tokenizer, output.output_ids[0]) for output in outputs]
        assert texts["item"] == texts["dispatch"] == ["catsat", "catsat"]

    def test_batched_prompt_that_extends_another_prompt_and_part_of_its_answer_follows_the_grammar(
        self, model, tokenizer,
    ):
        the, cat, sat, mat = tokenizer.convert_tokens_to_ids(["the", "cat", "sat", "mat"])
        pad = tokenizer.pad_token_id
        # the second prompt is the first prompt and the first two tokens of the answer it receives
        input_ids = torch.tensor([[pad, pad, 0, the, mat], [0, the, mat, cat, sat]])
        for seed_kwargs in ({}, {"seed": 1, "seed_scope": "dispatch"}):
            control = ConstrainedDecoding(choice=["catsatdog", "dogran"], include_in_scoring=False)
            pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
            pipeline.steer()
            bias = _ScriptedBias(lambda step, row, last: {cat: 50.0})
            outputs = pipeline.generate(
                input_ids=input_ids, attention_mask=(input_ids != pad).long(), max_new_tokens=8, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, logits_processor=[bias], return_output=True, **seed_kwargs,
            )
            texts = [_constrained_text(tokenizer, output.output_ids[0]) for output in outputs]
            assert texts == ["catsatdog", "catsatdog"], seed_kwargs

    def test_batching_follows_the_constraint_form(self):
        assert ConstrainedDecoding(regex="cat|dog").supports_batching is True
        assert ConstrainedDecoding(automaton=object()).supports_batching is False

    def test_generations_use_independent_automaton_state(self, tokenizer):
        control = ConstrainedDecoding(choice=["cat", "dog"], include_in_scoring=False)
        control.tokenizer = tokenizer
        prompt = torch.tensor([[0, 3, 4]])
        first = control.get_logits_processors(prompt, {})[0]
        second = control.get_logits_processors(prompt, {})[0]
        assert first.automaton is not second.automaton
        assert first.automaton._compiled is second.automaton._compiled


class TestXGrammarAutomaton:

    @pytest.fixture
    def automaton(self, tokenizer):
        return compile_constraint_automaton(ConstraintSource(kind="choice", value=("catsat", "dogran")), tokenizer)

    def test_rows_advance_independently(self, automaton, tokenizer):
        cat, dog, sat, ran = tokenizer.convert_tokens_to_ids(["cat", "dog", "sat", "ran"])
        prompt = torch.tensor([[0, 3, 7], [0, 3, 7]])
        automaton.reset(prompt)
        first, second = automaton.allowed(torch.cat([prompt, torch.tensor([[cat], [dog]])], dim=1))
        assert first.tolist() == [sat]
        assert second.tolist() == [ran]

    def test_out_of_grammar_token_leaves_only_stop_tokens(self, automaton, tokenizer):
        cat, sat, the = tokenizer.convert_tokens_to_ids(["cat", "sat", "the"])
        prompt = torch.tensor([[0, 3, 7], [0, 3, 7]])
        automaton.reset(prompt)
        # row 1 receives a spliced token the grammar rejects, as a `Fixed` phase would insert
        first, second = automaton.allowed(torch.cat([prompt, torch.tensor([[cat], [the]])], dim=1))
        assert first.tolist() == [sat]
        assert second.tolist() == [tokenizer.eos_token_id]

        processor = ConstraintProcessor(automaton)
        prefix = torch.cat([prompt, torch.tensor([[cat], [the]])], dim=1)
        out = processor(prefix, torch.zeros(2, len(tokenizer)))
        assert torch.isfinite(out[0]).nonzero().flatten().tolist() == [sat]
        assert torch.isfinite(out[1]).nonzero().flatten().tolist() == [tokenizer.eos_token_id]

    def test_rewind_replays_from_the_prompt_boundary(self, automaton, tokenizer):
        cat, dog, ran = tokenizer.convert_tokens_to_ids(["cat", "dog", "ran"])
        processor = ConstraintProcessor(automaton)
        prompt = torch.tensor([[0, 3, 7]])
        _permitted(processor, prompt)
        _permitted(processor, _extend(prompt, [[dog]]))
        _permitted(processor, _extend(prompt, [[dog, ran]]))
        # a rewind to a shorter prefix replays the row's output from the prompt boundary
        assert _permitted(processor, _extend(prompt, [[dog]])) == [[ran]]
        # a row that begins with no known prompt begins a new constrained region
        assert _permitted(processor, torch.tensor([[0, 5, 6]])) == [[cat, dog]]

    def test_rows_beyond_the_previous_batch_continue_their_prompt(self, automaton, tokenizer):
        cat, dog, sat, ran = tokenizer.convert_tokens_to_ids(["cat", "dog", "sat", "ran"])
        processor = ConstraintProcessor(automaton)
        prompt = torch.tensor([[0, 3, 7]])
        _permitted(processor, prompt)
        # two continuations of one prompt row, as when a driver expands a kept row into candidates
        assert _permitted(processor, _extend(prompt.expand(2, -1), [[cat], [dog]])) == [[sat], [ran]]

    def test_repacked_rows_keep_their_grammar_state(self, automaton, tokenizer):
        cat, dog, sat = tokenizer.convert_tokens_to_ids(["cat", "dog", "sat"])
        pad = tokenizer.pad_token_id
        processor = ConstraintProcessor(automaton)
        # the second prompt begins with the first one once the pads are removed
        prompts = torch.tensor([[pad, pad, 0, 3], [0, 3, 7, 8]])
        assert _permitted(processor, prompts) == [[cat, dog], [cat, dog]]
        assert _permitted(processor, _extend(prompts, [[cat], [cat]])) == [[sat], [sat]]
        # the rows re-packed with one more leading pad each, as a driver does for its next phase
        repacked = torch.tensor([[pad, pad, pad, 0, 3, cat], [pad, 0, 3, 7, 8, cat]])
        assert _permitted(processor, repacked) == [[sat], [sat]]

    def test_prompt_that_begins_with_an_earlier_prompt_starts_its_own_output(self, automaton, tokenizer):
        cat, dog, the = tokenizer.convert_tokens_to_ids(["cat", "dog", "the"])
        pad = tokenizer.pad_token_id
        # a later prompt that continues the first one, e.g., the next turn of the same conversation
        prompts = torch.tensor([[pad, pad, 0, 3, 7], [0, 3, 7, the, cat]])
        processor = ConstraintProcessor(automaton.fresh(prompts))
        prompt = torch.tensor([[0, 3, 7]])
        _permitted(processor, prompt)
        _permitted(processor, _extend(prompt, [[cat]]))
        assert _permitted(processor, prompts[1:]) == [[cat, dog]]

    def test_inferred_prompt_that_begins_with_an_earlier_prompt_starts_its_own_output(self, automaton, tokenizer):
        cat, dog, the = tokenizer.convert_tokens_to_ids(["cat", "dog", "the"])
        # no prompts passed to `fresh()`, as for a compiled automaton supplied to the control
        for unseeded in (automaton, automaton.fresh()):
            processor = ConstraintProcessor(unseeded)
            prompt = torch.tensor([[0, 3, 7]])
            _permitted(processor, prompt)
            _permitted(processor, _extend(prompt, [[cat]]))
            # a later prompt whose next token after the first prompt the grammar rejects
            assert _permitted(processor, _extend(prompt, [[the, cat]])) == [[cat, dog]]

    def test_batched_rows_that_reach_another_prompt_continue_their_output(self, automaton, tokenizer):
        cat, sat = tokenizer.convert_tokens_to_ids(["cat", "sat"])
        pad = tokenizer.pad_token_id
        eos = tokenizer.eos_token_id
        # the second prompt is the first prompt and part of an answer, which the first row then generates
        for answer, expected in (([cat, sat], [[eos], [eos]]), ([cat], [[sat], [sat]])):
            prompts = torch.tensor([[pad] * len(answer) + [0, 3, 7], [0, 3, 7, *answer]])
            processor = ConstraintProcessor(automaton.fresh(prompts))
            _permitted(processor, prompts)
            prefix = prompts
            for token_id in answer[:-1]:
                prefix = _extend(prefix, [[token_id], [token_id]])
                _permitted(processor, prefix)
            assert _permitted(processor, _extend(prefix, [answer[-1:], answer[-1:]])) == expected

    def test_prompt_that_extends_an_earlier_prompt_and_its_output_starts_its_own_output(self, automaton, tokenizer):
        the, cat, sat, mat, dog = tokenizer.convert_tokens_to_ids(["the", "cat", "sat", "mat", "dog"])
        pad = tokenizer.pad_token_id
        prompts = torch.tensor([[pad, pad, pad, pad, 0, the, mat], [0, the, mat, cat, sat, the, dog]])
        processor = ConstraintProcessor(automaton.fresh(prompts))
        first = torch.tensor([[0, the, mat]])
        _permitted(processor, first)
        _permitted(processor, _extend(first, [[cat]]))
        assert _permitted(processor, _extend(first, [[cat, sat]])) == [[tokenizer.eos_token_id]]
        # the second prompt decodes after the first, and begins with the first prompt and its output
        assert _permitted(processor, prompts[1:]) == [[cat, dog]]

    def test_candidates_decoded_one_at_a_time_continue_their_prompt(self, automaton, tokenizer):
        cat, dog, sat, ran = tokenizer.convert_tokens_to_ids(["cat", "dog", "sat", "ran"])
        prompt = torch.tensor([[0, 3, 7]])
        processor = ConstraintProcessor(automaton.fresh(prompt))
        # each candidate's one-token phase sees only the prompt, then its next phase continues after its token
        assert _permitted(processor, prompt) == [[cat, dog]]
        assert _permitted(processor, prompt) == [[cat, dog]]
        assert _permitted(processor, _extend(prompt, [[cat]])) == [[sat]]
        assert _permitted(processor, _extend(prompt, [[dog]])) == [[ran]]

    def test_tokens_spliced_before_the_first_generated_token_precede_the_output(self, automaton, tokenizer):
        cat, dog, sat, ran, the = tokenizer.convert_tokens_to_ids(["cat", "dog", "sat", "ran", "the"])
        prompt = torch.tensor([[0, 3, 7]])
        processor = ConstraintProcessor(automaton.fresh(prompt))
        # a prefix spliced onto the prompt before generation, as a routed prefix or a leading `Fixed` phase inserts
        spliced = _extend(prompt, [[the]])
        assert _permitted(processor, spliced) == [[cat, dog]]
        assert _permitted(processor, _extend(spliced, [[cat]])) == [[sat]]
        # a second candidate of the spliced prompt, decoded after the first
        assert _permitted(processor, spliced) == [[cat, dog]]
        assert _permitted(processor, _extend(spliced, [[dog]])) == [[ran]]

    def test_processor_rejects_a_row_count_mismatch(self):
        class _TwoRowAutomaton:
            def reset(self, prefix_ids):
                pass

            def allowed(self, prefix_ids):
                return [torch.tensor([3]), torch.tensor([4])]

        processor = ConstraintProcessor(_TwoRowAutomaton())
        with pytest.raises(ValueError, match="2 allowed sets for a batch of 1 rows"):
            processor(torch.tensor([[0, 3]]), torch.zeros(1, 6))


class TestSpipeRoundTrip:
    """Bundles saved from each constraint form load and instantiate without `allow_code`."""

    TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    def _saved(self, tmp_path, control, freeze):
        from steerability.spipe import SPipe

        spipe = SteeringPipeline(model_name_or_path=self.TINY_LLAMA, controls=[control]).to_spipe(freeze=freeze)
        return spipe, SPipe.load(spipe.save(tmp_path / "constrained.spipe"))

    @pytest.mark.parametrize("freeze", [False, True])
    @pytest.mark.parametrize("constraint", [
        {"choice": ["cat", "dog"]},
        {"regex": "cat|dog"},
        {"json_schema": {"type": "object", "properties": {"animal": {"type": "string"}}}},
        {"grammar": 'root ::= "cat" | "dog"'},
        {"source": ConstraintSource(kind="regex", value="cat|dog")},
    ], ids=["choice", "regex", "json_schema", "grammar", "source"])
    def test_constraint_forms_reload(self, tmp_path, constraint, freeze):
        original = ConstrainedDecoding(**constraint)
        spipe, loaded = self._saved(tmp_path, original, freeze)
        assert loaded.is_frozen is freeze
        (control,) = loaded.pipeline().output_controls
        assert control.args == original.args
        (inspected,) = loaded.instantiate_entry(0, lenient=True)
        assert inspected.source == original.source
        assert loaded.config_id == spipe.config_id

    def test_a_source_that_disagrees_with_the_convenience_field_raises(self, tmp_path):
        from steerability.spipe import SPipe
        from steerability.spipe.errors import SpipeFormatError

        spipe, _ = self._saved(tmp_path, ConstrainedDecoding(regex="cat|dog"), freeze=False)
        manifest = spipe.manifest
        manifest["controls"][0]["args"]["source"]["fields"]["value"] = "cat"
        edited = SPipe(manifest, store=None, base_dir=None, allow_code=False)
        with pytest.raises(SpipeFormatError, match="exactly one constraint"):
            edited.instantiate_entry(0)
        with pytest.raises(ValueError, match="exactly one constraint"):
            ConstrainedDecoding(source=ConstraintSource(kind="regex", value="cat"), regex="cat|dog")
