"""Shared setup for the experiment scripts: model choice, subsets, and run names."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from inklab.constants import LOGS_DIR, MAIN_MMPROJ, MAIN_MODEL, RUNS_DIR, VARLOOP_MMPROJ, VARLOOP_MODEL
from inklab.server import ServerConfig


class ModelChoice(Enum):
    """The two checkpoints of the release."""

    MAIN = "main"
    VARLOOP = "varloop"

    def files(self) -> tuple[Path, Path]:
        """Return (model, mmproj) paths."""
        if self is ModelChoice.MAIN:
            return MAIN_MODEL, MAIN_MMPROJ
        return VARLOOP_MODEL, VARLOOP_MMPROJ


@dataclass(frozen=True)
class RunSpec:
    """One experiment configuration and where its files go."""

    task: str
    model: ModelChoice
    passes: int | None
    effort: str
    extra: str = ""

    @property
    def name(self) -> str:
        """Return a file-safe run name."""
        passes = f"p{self.passes}" if self.passes is not None else "pdefault"
        parts = [self.task, self.model.value, passes, self.effort] + ([self.extra] if self.extra else [])
        return "_".join(parts)

    @property
    def output(self) -> Path:
        """Return the JSON Lines result file."""
        return RUNS_DIR / f"{self.name}.jsonl"

    @property
    def server_log(self) -> Path:
        """Return the server log file."""
        return LOGS_DIR / f"server_{self.name}.log"

    def server_config(self, with_vision: bool, n_cpu_moe: int, parallel: int) -> ServerConfig:
        """Return the server settings for this run."""
        model, mmproj = self.model.files()
        return ServerConfig(model=model, mmproj=mmproj if with_vision else None,
                            n_cpu_moe=n_cpu_moe, parallel=parallel, loop_passes=self.passes)


def balanced_subset(labels: np.ndarray, per_class: int) -> list[int]:
    """Return the first per_class indices of every class, sorted.

    Raises:
        ValueError: If a class has fewer than per_class examples.
    """
    chosen: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < per_class:
            raise ValueError(f"class {label} has only {len(indices)} examples")
        chosen.extend(int(i) for i in indices[:per_class])
    return sorted(chosen)
