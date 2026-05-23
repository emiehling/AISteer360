"""RetrievalMemory: corpus + retrieval artifacts for EPR."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import joblib
import numpy as np


@dataclass
class RetrievalMemory:
    """Memory for retrieval-based input augmentation.

    Holds the corpus of demonstrations plus the artifacts needed to retrieve from it at serve time. Contents depend on
    mode:

      - `"bm25"`: corpus + serialized TF-IDF/BM25 state.
      - `"dense"`: corpus + dense embeddings under an off-the-shelf encoder + the encoder's name/path (for re-loading
        the query-side encoder at serve time).
      - `"epr"`: corpus + dense embeddings under the trained prompt encoder + paths to the saved input and prompt
        encoder directories.

    Fifth canonical Memory shape after `TextMemory`, `RuleStreamMemory`, `CausalPoolMemory`, `ModelMemory`. Used only
    by EPR at Phase 8; may be promoted to `common/memory/` if a second method adopts a similar shape.

    Attributes:
        corpus: Demonstration pairs. Each dict has `"input"` and `"output"` keys.
        mode: `"bm25"`, `"dense"`, or `"epr"`.
        dense_embeddings: `[N, dim]` float32 array of corpus embeddings (dense / epr modes).
        input_encoder_name_or_path: Identifier or local path for re-loading the input-side encoder.
        prompt_encoder_name_or_path: Identifier or local path for re-loading the prompt-side encoder.
        encoder_pooling: `"cls"` or `"mean"`. Should match how embeddings were computed.
        bm25_state: Serialized TF-IDF index parameters (bm25 mode).
        demo_template: Default demonstration template captured at training time.
        demo_separator: Default separator captured at training time.
    """

    corpus: list[dict]
    mode: Literal["bm25", "dense", "epr"]

    dense_embeddings: np.ndarray | None = None

    input_encoder_name_or_path: str | None = None
    prompt_encoder_name_or_path: str | None = None
    encoder_pooling: Literal["cls", "mean"] = "cls"

    bm25_state: dict | None = None

    demo_template: str = "Input: {input}\nOutput: {output}"
    demo_separator: str = "\n\n"

    model_type: str = field(default="retrieval", init=False)

    _EXTENSION = ".rmem"

    def save(self, path: str) -> None:
        """Save to a directory `<path>.rmem/`.

        Layout:

          - `meta.json` — `model_type`, `mode`, encoder identifiers, templates.
          - `corpus.jsonl` — one example per line.
          - `embeddings.npy` — dense / epr modes.
          - `bm25_state.joblib` — bm25 mode (sklearn `TfidfVectorizer` + sparse doc matrix; pickled via joblib).
          - `input_encoder/`, `prompt_encoder/` — epr mode only (HF `save_pretrained` directories), if the in-memory
            encoder identifiers point to subdirectories of this `.rmem` folder. Otherwise, the path is recorded in
            `meta.json` only.

        Args:
            path: Output path (directory). `.rmem` extension appended if not present.
        """
        if not path.endswith(self._EXTENSION):
            path += self._EXTENSION
        os.makedirs(path, exist_ok=True)

        meta = {
            "model_type": self.model_type,
            "mode": self.mode,
            "input_encoder_name_or_path": self.input_encoder_name_or_path,
            "prompt_encoder_name_or_path": self.prompt_encoder_name_or_path,
            "encoder_pooling": self.encoder_pooling,
            "demo_template": self.demo_template,
            "demo_separator": self.demo_separator,
            "corpus_size": len(self.corpus),
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        with open(os.path.join(path, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for row in self.corpus:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if self.dense_embeddings is not None:
            np.save(os.path.join(path, "embeddings.npy"), self.dense_embeddings)

        if self.mode == "bm25":
            if self.bm25_state is None:
                raise RuntimeError("RetrievalMemory.save: mode='bm25' but bm25_state is None.")
            joblib.dump(self.bm25_state, os.path.join(path, "bm25_state.joblib"))

    @classmethod
    def load(cls, path: str) -> "RetrievalMemory":
        """Load a `RetrievalMemory` from a directory.

        Note: for `mode="bm25"`, `bm25_state.joblib` is unpickled via `joblib.load`, which executes arbitrary Python on
        a malicious artifact. Only load files from trusted sources.

        Encoders are NOT re-instantiated at load time; the consuming `EPR` control loads them via the stored paths.

        Args:
            path: Directory path. `.rmem` extension appended if not present.

        Returns:
            Loaded `RetrievalMemory` instance.

        Raises:
            ValueError: If the meta `model_type` does not match this class.
        """
        if not path.endswith(cls._EXTENSION):
            path += cls._EXTENSION

        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model_type") != "retrieval":
            raise ValueError(
                f"Cannot load RetrievalMemory: meta model_type is "
                f"{meta.get('model_type')!r}, expected 'retrieval'."
            )

        mode = meta["mode"]

        corpus: list[dict] = []
        with open(os.path.join(path, "corpus.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                corpus.append(json.loads(line))

        embeddings_path = os.path.join(path, "embeddings.npy")
        dense_embeddings = np.load(embeddings_path) if os.path.exists(embeddings_path) else None

        bm25_state = None
        if mode == "bm25":
            bm25_state = joblib.load(os.path.join(path, "bm25_state.joblib"))

        return cls(
            corpus=corpus,
            mode=mode,
            dense_embeddings=dense_embeddings,
            input_encoder_name_or_path=meta.get("input_encoder_name_or_path"),
            prompt_encoder_name_or_path=meta.get("prompt_encoder_name_or_path"),
            encoder_pooling=meta.get("encoder_pooling", "cls"),
            bm25_state=bm25_state,
            demo_template=meta.get("demo_template", "Input: {input}\nOutput: {output}"),
            demo_separator=meta.get("demo_separator", "\n\n"),
        )
