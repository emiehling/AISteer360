"""Result types for each stage of the calibration builder."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aisteer360.algorithms.state_control._common.specs import ContrastivePairs


@dataclass
class GenerationResult:
    """Output of the contrastive pair generation stage.

    Attributes:
        pairs: The `ContrastivePairs` object (reuses the existing type directly).
        seed_prompts_used: The actual seed prompts that were used.
        config: The serialized `GenerationConfig` that produced this result.
    """

    pairs: ContrastivePairs
    seed_prompts_used: list[str]
    config: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Write pairs to a JSONL file (one record per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prompts = self.pairs.prompts or [""] * len(self.pairs.positives)
        behavior = self.config.get("behavior", "") if self.config else ""
        with open(path, "w") as f:
            for prompt, pos, neg in zip(prompts, self.pairs.positives, self.pairs.negatives):
                record = {
                    "prompt": prompt,
                    "positive": pos,
                    "negative": neg,
                    "behavior": behavior,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "GenerationResult":
        """Load a GenerationResult from a JSONL file.

        Malformed trailing lines (e.g. from an interrupted write) are treated as truncation and discarded.
        """
        path = Path(path)
        records: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        pairs = ContrastivePairs(
            positives=[r["positive"] for r in records],
            negatives=[r["negative"] for r in records],
            prompts=[r["prompt"] for r in records],
        )
        return cls(
            pairs=pairs,
            seed_prompts_used=[r["prompt"] for r in records],
            config={"behavior": records[0]["behavior"]} if records else {},
        )


# extraction stage has no new result type; it returns a SteeringVector directly.


@dataclass
class CellResult:
    """Evaluation result for a single (layer, multiplier) cell.

    Attributes:
        layer: Layer index where the vector was applied.
        multiplier: Scaling factor used.
        score_mean: Mean judge score across eval prompts.
        score_delta: Improvement over baseline (`score_mean - baseline_score`).
        coherence: Self-consistency score (0 to 1).
        perplexity: Mean perplexity of steered generations.
        perplexity_delta: Perplexity change from baseline.
        coherent: Whether this cell passes the quality gate.
        generations: Optional list of per-prompt generation details (prompt, steered_text, judge_score,
            judge_reason).
    """

    layer: int
    multiplier: float
    score_mean: float
    score_delta: float
    coherence: float
    perplexity: float
    perplexity_delta: float
    coherent: bool
    generations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CalibrationResult:
    """Full output of the calibration sweep.

    Attributes:
        cells: All evaluated cells.
        baseline_score: Mean judge score with no steering (multiplier=0).
        baseline_perplexity: Mean perplexity with no steering.
        peak_cell: The coherent cell with the highest `score_delta` (None when no cell passed the gate).
        grid_shape: `(n_layers, n_multipliers)` for reconstructing the heatmap.
        layers: Ordered list of layer indices in the grid.
        multipliers: Ordered list of multiplier values in the grid.
        config: The serialized `CalibrationConfig` that produced this result.
    """

    cells: list[CellResult]
    baseline_score: float
    baseline_perplexity: float
    peak_cell: CellResult | None
    grid_shape: tuple[int, int]
    layers: list[int]
    multipliers: list[float]
    config: dict = field(default_factory=dict)

    def to_heatmap_grid(self) -> dict[str, list[list[float | None]]]:
        """Reshape cells into 2D grids keyed by metric name.

        Returns:
            A dict with keys `"score_delta"`, `"coherence"`, and `"perplexity"`, each mapping to a
            `[n_layers x n_multipliers]` nested list. Missing cells are `None`.
        """
        lookup = {(c.layer, c.multiplier): c for c in self.cells}
        grids: dict[str, list[list[float | None]]] = {
            "score_delta": [],
            "coherence": [],
            "perplexity": [],
        }
        for layer in self.layers:
            row_sd: list[float | None] = []
            row_co: list[float | None] = []
            row_pp: list[float | None] = []
            for mult in self.multipliers:
                cell = lookup.get((layer, mult))
                row_sd.append(cell.score_delta if cell else None)
                row_co.append(cell.coherence if cell else None)
                row_pp.append(cell.perplexity if cell else None)
            grids["score_delta"].append(row_sd)
            grids["coherence"].append(row_co)
            grids["perplexity"].append(row_pp)
        return grids

    def save(self, path: str | Path) -> None:
        """Serialize the calibration result (without the bulky generations) to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cells": [
                {
                    **{k: v for k, v in asdict(c).items() if k != "generations"},
                    "n_generations": len(c.generations),
                }
                for c in self.cells
            ],
            "baseline_score": self.baseline_score,
            "baseline_perplexity": self.baseline_perplexity,
            "peak_cell": (
                {k: v for k, v in asdict(self.peak_cell).items() if k != "generations"}
                if self.peak_cell
                else None
            ),
            "grid_shape": list(self.grid_shape),
            "layers": self.layers,
            "multipliers": self.multipliers,
            "config": self.config,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        """Reconstruct a `CalibrationResult` from a saved JSON file."""
        data = json.loads(Path(path).read_text())
        cells = [
            CellResult(**{k: v for k, v in c.items() if k != "n_generations"})
            for c in data["cells"]
        ]
        peak_data = data.get("peak_cell")
        peak = (
            CellResult(**{k: v for k, v in peak_data.items() if k != "n_generations"})
            if peak_data
            else None
        )
        return cls(
            cells=cells,
            baseline_score=data["baseline_score"],
            baseline_perplexity=data["baseline_perplexity"],
            peak_cell=peak,
            grid_shape=tuple(data["grid_shape"]),
            layers=data["layers"],
            multipliers=data["multipliers"],
            config=data.get("config", {}),
        )
