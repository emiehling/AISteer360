import shutil
import tempfile
from pathlib import Path

from steerability.utils.optional import require

require("mergekit")
import mergekit.config as mk_config
import mergekit.merge as mk_merge
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import CheckpointArtifact
from steerability.algorithms.core.fits import active_fit_cache, product_is_stored, resolve_product, stored_product
from steerability.algorithms.structural_control.base import StructuralControl
from steerability.algorithms.structural_control.wrappers.mergekit.args import MergeKitArgs


class MergeKit(StructuralControl):
    """
    Wrapper for merging models via MergeKit [https://github.com/arcee-ai/mergekit](https://github.com/arcee-ai/mergekit).

    MergeKit combines multiple language models using various merge strategies like linear interpolation, SLERP, and
    TIES. This wrapper integrates MergeKit's functionality to enable structural control through model composition.

    The process involves loading a merge configuration (from YAML or dict), executing the merge operation, and
    optionally loading the resulting merged model. Without a fit cache, an existing `out_path` is reused unless
    `force_remerge` is set. Under an active fit cache, the merged checkpoint is the control's one fit. A stored
    checkpoint is installed as the frozen form does (a `load_checkpoint` steer), and otherwise the merge runs into a
    directory the control creates for that run, whose checkpoint is stored and then installed. An existing `out_path`
    never becomes the product of a fit.

    Reference:

    - "Arcee's MergeKit: A Toolkit for Merging Large Language Models"
      Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vladimir Karpukhin, Brian Benedict,
      Mark McQuade, Jacob Solawetz
      [https://aclanthology.org/2024.emnlp-industry.36](https://aclanthology.org/2024.emnlp-industry.36)
    """

    Args = MergeKitArgs

    _product: CheckpointArtifact | None = None

    def artifact_capability(self) -> Capability:
        """Merging always leaves a full-weights checkpoint."""
        return Capability.SERVE_CHECKPOINT

    def export_artifact(self) -> CheckpointArtifact:
        """The merged checkpoint directory.

        After `steer()` under a fit cache, and before it while the active cache contains the product, this is the stored
        checkpoint. Otherwise it is `out_path`.
        """
        if self._product is None and product_is_stored(control=self):
            self._product = CheckpointArtifact(path=str(stored_product(control=self)["artifact"]))
        if self._product is not None:
            return self._product
        return CheckpointArtifact(path=str(self.args.out_path))

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.FACTS` while the active fit cache contains the merged checkpoint, else `ModelAccess.MODULE`."""
        return ModelAccess.FACTS if product_is_stored(control=self) else ModelAccess.MODULE

    def steer_fits(self) -> tuple[tuple[str, str], ...]:
        """One `("MergedWeights", "direction")` fit, or none while the active fit cache contains its checkpoint."""
        if product_is_stored(control=self):
            return ()
        return (("MergedWeights", "direction"),)

    def export_state(self) -> dict:
        """The merged checkpoint (`export_artifact()`) under the `"artifact"` key, for freezing and the fit cache."""
        return {"artifact": self.export_artifact()}

    def frozen_form(self, state: dict) -> tuple[str, dict]:
        """The merged checkpoint as a `load_checkpoint` entry."""
        return "structural_control/load_checkpoint", {"path": state["artifact"]}

    def fit_identity(self) -> dict:
        """The merge-relevant args projection: the merge configuration and the dtype."""
        return {
            "config_path": str(self.args.config_path) if self.args.config_path else None,
            "config_dict": self.args.config_dict,
            "dtype": self.args.dtype,
        }

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase = None,
            session=None,
            **_
    ) -> PreTrainedModel:
        """Execute model merging via MergeKit and optionally return the merged model.

        Under an active fit cache, the product is resolved through the cache (`resolve_product`). The merge runs only
        when the cache does not contain the checkpoint, into a directory the control creates for that run and removes
        once the checkpoint is stored. The stored checkpoint is then installed as the frozen form does (when
        `load_merged`), and the control refers to it (`export_artifact`). The steps below are those of a steer without
        a fit cache.

        Performs structural steering by merging multiple models according to a configuration file or dictionary.
        Supports caching to avoid redundant merge operations and can either return the merged model or the original
        model based on configuration.

        The method follows this logic:

        1. Load merge configuration from YAML file or dictionary
        2. Check if merged model already exists (skip if `force_remerge=False`)
        3. Execute merge if needed using MergeKit
        4. Optionally load and return the merged model

        Args:
            model (PreTrainedModel): The base model (potentially unused depending on the method).
            tokenizer (PreTrainedTokenizerBase, optional): Base tokenizer (currently unused).

        Returns:
            PreTrainedModel: Either the merged model (if `load_merged=True`) or the original model. When returning
            merged model, attempts to attach a new tokenizer if one was created during merging.

        Note:

        - If out_path exists and `force_remerge=False`, skips merging and loads cached result
        - Merged model saved to `out_path` directory with full weights and config
        - If `load_merged=False`, performs merge but returns original model
        """
        args: MergeKitArgs = self.args

        if args.config_path:
            with open(args.config_path, "r", encoding="utf-8") as config_file:
                config = mk_config.MergeConfiguration.model_validate(yaml.safe_load(config_file))
        else:
            config = mk_config.MergeConfiguration.model_validate(args.config_dict)

        if active_fit_cache() is not None:
            return self._steer_product(model=model, tokenizer=tokenizer, session=session, config=config)

        # find merged weights
        out_path = Path(args.out_path)
        if out_path.exists() and not args.force_remerge:
            if args.load_merged:
                merged = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=str(out_path),
                    device_map=args.device_map,
                    trust_remote_code=args.trust_remote_code,
                    dtype=getattr(torch, args.dtype)
                )
                return merged
            return model

        # merge
        self._run_merge(config=config, out_path=out_path)

        # load merged checkpoint (and check if merge returned new tokenizer)
        if args.load_merged:
            merged = AutoModelForCausalLM.from_pretrained(
                out_path,
                dtype=getattr(torch, args.dtype),
                device_map=args.device_map,
                trust_remote_code=args.trust_remote_code,
            )
            try:
                merged.tokenizer = AutoTokenizer.from_pretrained(
                    out_path,
                    trust_remote_code=args.trust_remote_code
                )
            except Exception:
                pass
            return merged

        return model

    def _steer_product(self, *, model, tokenizer, session, config) -> PreTrainedModel | None:
        """Resolve the merged checkpoint through the active fit cache and install it."""
        if self._product is None:
            created: list[Path] = []

            def fit() -> dict:
                run_dir = Path(tempfile.mkdtemp(prefix="steerability-merge-"))
                created.append(run_dir)
                self._run_merge(config=config, out_path=run_dir)
                return {"artifact": CheckpointArtifact(path=str(run_dir))}

            try:
                artifact = resolve_product(control=self, fit=fit)["artifact"]
                self._product = CheckpointArtifact(path=str(getattr(artifact, "path", artifact)))
            finally:
                for run_dir in created:
                    if self._product is None or Path(self._product.path) != run_dir:
                        shutil.rmtree(run_dir, ignore_errors=True)
        if not self.args.load_merged:
            return model
        from steerability.algorithms.core.registry import resolve_method_key

        method, loader_args = self.frozen_form(state={"artifact": self._product.path})
        loader = resolve_method_key(key=method).control_cls(**loader_args)
        return loader.steer(model=model, tokenizer=tokenizer, session=session)

    def _run_merge(self, *, config, out_path: Path) -> None:
        mk_merge.run_merge(
            merge_config=config,
            out_path=str(out_path),
            options=mk_merge.MergeOptions(
                cuda=self.args.allow_cuda,
                trust_remote_code=self.args.trust_remote_code,
                **self.args.extra_merge_options,
            ),
        )
