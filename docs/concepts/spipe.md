# Sharing pipelines (`.spipe`)

A steered pipeline normally exists only as in-memory Python objects. The `.spipe` format writes a pipeline down as a
portable bundle that can be saved, version-controlled, and handed to another person or machine.

One format contains two layers of information:

- **The recipe**: the model reference plus the pipeline's `Recipe`, i.e., one entry per enabled control with its
  method key and constructor arguments, and the recipe's id. Every `.spipe` contains the recipe. Loading a recipe-only
  bundle and calling `steer()` re-runs any fits and training. Results may differ across machines because of GPU
  nondeterminism.
- **The frozen resolution**: what the steer step actually produced, i.e., fitted steering vectors, probes, LoRA
  adapters, and optimized prompts, stored content-addressed alongside the recipe. Each artifact records the fit digest
  of the control that produced it, and a lock section records the protocol the bundle was fitted under and the facts
  observed on its model. Loading a frozen bundle yields controls in precomputed form, where `steer()` still runs but
  is cheap and model-free.

Freezing is a rewrite of the recipe rather than a second format. A frozen entry is an ordinary control constructed
with precomputed arguments: a CAA fitted from data freezes as a CAA constructed with the fitted vector, and a
fine-tune freezes as a [`LoadLoRA`](../reference/algorithms/structural_control/load_lora.md) or
[`LoadCheckpoint`](../reference/algorithms/structural_control/load_checkpoint.md) pointing at the trained product.
Loading therefore takes the same construction and `steer()` path as any hand-built pipeline.

## Saving and loading

```python
pipeline.steer()
spipe = pipeline.to_spipe()          # frozen by default once steered
spipe.save("formal_tone.spipe")      # a .spipe path writes a zip (any other path writes a directory)
```

```python
from steerability.spipe import SPipe

spipe = SPipe.load("formal_tone.spipe")
pipeline = spipe.pipeline()          # backend, device, and dtype stay the caller's choice
pipeline.steer()                     # installs the frozen artifacts and fits nothing
response = pipeline.generate(...)
```

`to_spipe(freeze=False)` forces a recipe-only bundle from a steered pipeline, and `spipe.thaw()` turns a frozen bundle
back into its recipe. `save(..., artifacts="thin")` writes the manifest without the artifact payloads. A thin bundle
loads against an external store via `SPipe.load(path, artifact_store=...)`. A model revision pinned through
`hf_model_kwargs["revision"]` is recorded in the manifest, and `spipe.pipeline()` loads that revision unless the
caller's `hf_model_kwargs` sets one.

### Freezing without a model

A pipeline with nothing to fit can be frozen without loading the model. `to_spipe(freeze=True)` on an unsteered
pipeline succeeds when every enabled control is recipe-frozen, i.e., its steer step declares only `facts` access,
declares no fits, and exports no state. A prompt-only pipeline (`SystemPrompt`, `UserPrefix`, a `FewShot` with fixed
pools, or a `PRewrite`, `GEPA`, or `CPO` constructed with `memory=`) qualifies, since its recipe already is its frozen
form. A control that blocks the freeze is named in the error, with the reason.

The lock of such a bundle records the protocol as usual, and its observed facts contain no model fingerprint and no
dtype, since no model was present. A consumer that treats the lock as the guarantee that nothing fits at load time
therefore gets that guarantee without either side loading weights, and the bundle steers cheaply on load.

## Staleness

Each frozen artifact records the fit digest of its control: the content id of the digests of the fits the control
declares, in order. These are the digests the fit cache keys fits by (see `fit_digests` in
`steerability.algorithms.core.fits`), so they cover the inputs a re-fit would consume and nothing else. Editing an inert
application parameter in the manifest (say, a CAA multiplier) leaves the pinned vector valid. Editing the training
data does not, and loading then fails with a staleness error that identifies the control and the fix (`thaw()` and
re-steer, or `allow_stale=True`).

## Verification

`spipe.verify()` reports on a bundle without loading a model: format validity, artifact integrity, staleness, version
compatibility, and whether the bundle references code. The staleness check decodes the recipe of each entry whose
frozen artifacts record a digest. An entry whose recipe does not decode in this process (e.g., a method key that is
not registered) is reported with a warning that its staleness check could not run. When the recorded or the running
toolkit version is `"unknown"` (the package is not installed), the versions are not compared. At `steer()` time,
frozen steering artifacts are checked against the model they are being installed on, under a policy chosen at
`pipeline(verify=...)`:

- `"strict"` (default): a wrong architecture or width is an error. A calibrated artifact (a probe, a gate threshold)
  on a model with a different weight fingerprint is an error. A direction artifact on different weights of the same
  architecture is a warning, since a direction can be transferred across fine-tunes deliberately.
- `"warn"`: every mismatch is a warning.
- `"off"`: no checks.

On an engine backend the check compares the recorded model reference with the served model reference instead of the
weight fingerprints, since the model weights are not readable there.

## Trust and `allow_code`

A `.spipe` from someone else is untrusted input. Loading never unpickles by default. Tensors are stored only as
safetensors, archives are extracted behind zip-safety guards (symlinks are rejected, both as archive members and as
artifact entries), and every artifact is verified against its content hash (an id of the form `sha256:` followed by 64
lowercase hex digits). A store sidecar whose type or encoding disagrees with the manifest's artifact record raises
`SpipeIntegrityError`. Three things require an explicit `allow_code=True` at load, similar to `trust_remote_code`:

- References to Python callables (`$ref`), e.g., a scorer function a prompt optimizer was configured with. The
  manifest's `code_dependent` flag says up front whether a bundle needs this, and the referenced modules must be on the
  import path.
- Dataclasses (`$dc`) other than the toolkit dataclasses whose construction only validates their fields
  (`codec.DECODABLE_DATACLASSES`), since decoding a dataclass calls its constructor. Toolkit enums decode without it,
  and `$dc` never constructs anything other than a dataclass or an enum.
- Artifact payloads that contain pickled files, since unpickling executes code. These include CPO's trained scorer
  memory, `PoolMemory` payloads, and any adapter or checkpoint directory with pickle-format files (e.g., `.pkl`, `.pt`,
  `.pth`, `.ckpt`, or a `.bin` file that is a zip archive or a pickle stream). Freezing leaves TRL trainer state
  (`training_args.bin`, optimizer, scheduler, scaler, and RNG state) out of adapter and checkpoint directories, and
  logs a warning when a frozen directory still contains pickled files.

Note that `allow_code` governs decoding only. Recipe args such as `trust_remote_code` take effect when the pipeline is
steered, which means that an untrusted bundle should be steered in a sandbox.

Frozen prompt-optimization bundles keep their search-only arguments (scorers, budgets) for provenance. A bundle whose
optimizer used a custom scorer is therefore code-dependent even though the frozen memory never calls it.

A bundle can also name a method defined outside the toolkit tree, registered with `register_method` (see
[adding your own steering method](../tutorials/add_new_steering_method.md#controls-outside-the-toolkit-tree)). Since
registration happens at import time, the defining package must be imported before the bundle's controls are
instantiated. Without it the bundle still loads, and instantiating that entry (or reading `spipe.recipe`) fails with
an error that points at `register_method`.

## Inspecting entries

The `entries` property lists the manifest's control entries as `SpipeEntry` records (`index`, `method`, the encoded
`args`, and whether a `resolved` section exists) without decoding them. Disabled controls are not written, so every
entry is enabled. Each entry and each resolved item is recipe-entry JSON (`method` and `args`), and instantiating it
goes through `Recipe.from_json(...).instantiate()`. The
`instantiate_entry(index, prefer=, verify=, lenient=)` method returns the control(s) of one entry through the same
decoding path that `pipeline()` uses. Errors are raised per entry. An unregistered method key, a malformed value, or
args the constructor rejects raise `SpipeFormatError`, and an entry that needs code raises `SpipeCodeRefError`.
With `lenient=True`, no bundle code is imported or run (`$ref` values become inert markers, `$data` stays an unloaded
`DataRef`, and no dataset loads). This allows each entry's base class, `steer_access()`, and `requirements()` to be
inspected before the bundle is steered:

```python
from steerability.spipe import SPipe, SpipeError

spipe = SPipe.load("submission.spipe")
for entry in spipe.entries:
    try:
        controls = spipe.instantiate_entry(entry.index, lenient=True)
    except SpipeError as error:  # e.g., an unregistered method key, or an entry that needs code
        print(entry.method, error)
        continue
    print(entry.method, [control.steer_access() for control in controls])
```

Note that an entry that needs code (e.g., a `$dc` class outside `codec.DECODABLE_DATACLASSES`, or a pickle-bearing
artifact) raises `SpipeCodeRefError` under `lenient=True`, even when the bundle was loaded with `allow_code=True`.

## Identity

A bundle's `recipe_id` is the id of the recipe the pipeline was built from (`Recipe.from_controls(...).id`), the same
id that `SteeringEval` records for a point and the store keys fits, rollouts, and evidence under. It is computed from
each method's `Args` dataclass, so it does not depend on key order, explicit defaults, or the order of entries across
categories, and it does not include the model reference. `spipe.recipe` reads the entries back as a `Recipe` whose id
equals `recipe_id`.

The model reference and the lock answer two different questions. `model.ref` says where a loader builds the model
from, and `to_spipe(model_ref=...)` overrides it. The lock's `protocol` (read back as a `Protocol` through
`spipe.protocol`) says what the bundle was fitted or measured on: the model and its revision, the tokenizer, load
arguments, backend, fit policy, decoding defaults, and package versions, with credentials left out. The two differ
only when `model_ref` was passed. The lock also records the facts observed on the loaded model (`facts`: model and
tokenizer fingerprints, chat-template digest, dtype, resolved commit) and an optional `certificate`, a mapping whose
`schema` is `"certificate/1"`. A bundle that is not frozen has a null lock, and its `protocol` is None.
