"""Summaries of the run files: accuracy, token use, speed, expert routing, and loop gates."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inklab.probe_data import CORE_END, CORE_START, N_PHYS, ProbeDump, layer_pass_to_slot
from inklab.runner import read_jsonl

N_EXPERTS = 128


@dataclass(frozen=True)
class AccuracySummary:
    """Accuracy and cost of one run.

    Attributes:
        name: Run name.
        n: Number of records.
        accuracy: Fraction correct.
        stderr: Binomial standard error of the accuracy.
        unparsed: Fraction of answers without a usable class or number.
        mean_tokens: Mean generated tokens per request, reasoning included.
        mean_seconds: Mean wall-clock seconds per request.
    """

    name: str
    n: int
    accuracy: float
    stderr: float
    unparsed: float
    mean_tokens: float
    mean_seconds: float


class RunSummarizer:
    """Turn a JSON Lines run file into an AccuracySummary."""

    def cifar(self, path: Path) -> AccuracySummary:
        """Summarize a CIFAR run; a record is correct when predicted equals label."""
        records = read_jsonl(path)
        correct = np.array([r["predicted"] == r["label"] for r in records], dtype=float)
        unparsed = np.array([r["predicted"] < 0 for r in records], dtype=float)
        return self._summary(path, records, correct, unparsed)

    def gsm8k(self, path: Path) -> AccuracySummary:
        """Summarize a GSM8K run; a record is correct when the last number matches."""
        records = read_jsonl(path)
        correct = np.array([r["correct"] for r in records], dtype=float)
        unparsed = np.array([r["extracted"] is None for r in records], dtype=float)
        return self._summary(path, records, correct, unparsed)

    @staticmethod
    def _summary(path: Path, records: list[dict], correct: np.ndarray, unparsed: np.ndarray) -> AccuracySummary:
        n = len(records)
        accuracy = float(correct.mean())
        return AccuracySummary(
            name=path.stem, n=n, accuracy=accuracy, stderr=float(np.sqrt(accuracy * (1 - accuracy) / n)),
            unparsed=float(unparsed.mean()),
            mean_tokens=float(np.mean([r["completion_tokens"] for r in records])),
            mean_seconds=float(np.mean([r["seconds"] for r in records])),
        )


def confusion_matrix(path: Path, n_classes: int) -> np.ndarray:
    """Return counts[true, predicted] for a CIFAR run; unparsed answers go to an extra last column."""
    counts = np.zeros((n_classes, n_classes + 1), dtype=int)
    for record in read_jsonl(path):
        predicted = record["predicted"] if record["predicted"] >= 0 else n_classes
        counts[record["label"], predicted] += 1
    return counts


class RoutingAnalysis:
    """Compare which experts the router picks for different kinds of tokens."""

    def __init__(self, dumps: list[tuple[str, ProbeDump]]) -> None:
        self.dumps = dumps

    def usage(self, category: str, slot: int) -> np.ndarray:
        """Return the share of routing picks per expert for one category at one slot."""
        counts = np.zeros(N_EXPERTS)
        for cat, dump in self.dumps:
            if cat != category:
                continue
            picks = dump.topk(slot)[self._mask(cat, dump)]
            np.add.at(counts, picks.reshape(-1), 1)
        return counts / max(counts.sum(), 1)

    def divergence(self, category_a: str, category_b: str, slot: int) -> float:
        """Return the Jensen-Shannon divergence (bits) between two categories' expert usage at one slot."""
        p, q = self.usage(category_a, slot), self.usage(category_b, slot)
        m = (p + q) / 2
        return float((self._kl(p, m) + self._kl(q, m)) / 2)

    def pass_overlap(self, layer: int, first: int, second: int, category: str) -> float:
        """Return the mean share of a token's 8 experts that stay the same between two passes of a core layer."""
        shares = []
        for cat, dump in self.dumps:
            if cat != category:
                continue
            mask = self._mask(cat, dump)
            a = dump.topk(layer_pass_to_slot(layer, first))[mask]
            b = dump.topk(layer_pass_to_slot(layer, second))[mask]
            shares += [len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]
        return float(np.mean(shares))

    @staticmethod
    def _mask(category: str, dump: ProbeDump) -> np.ndarray:
        # Image prompts are measured on their image tokens, text prompts on the user's own text tokens.
        return dump.is_image if category == "image" else dump.is_user_text

    @staticmethod
    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        keep = p > 0
        return float(np.sum(p[keep] * np.log2(p[keep] / q[keep])))


class LoopAnalysis:
    """Measure what the second and third pass of the loop core do."""

    def __init__(self, dumps: list[tuple[str, ProbeDump]]) -> None:
        self.dumps = dumps

    def gate_strength(self, kind: str, ctrl: str, layer: int, pass_number: int, category: str) -> float:
        """Return the mean |gate| over tokens of one category (gain or forget, tanh output)."""
        values = []
        for cat, dump in self.dumps:
            if cat == category:
                gate = dump.gate(kind, ctrl, layer_pass_to_slot(layer, pass_number))
                values.append(gate[RoutingAnalysis._mask(cat, dump), 1])
        return float(np.mean(np.concatenate(values)))

    def pass_change(self, first: int, second: int, category: str) -> dict[str, float]:
        """Compare the core output (layer 17) of two passes over the tokens of one category.

        Returns:
            Median relative L2 change, median cosine similarity, and median norm ratio (second / first).
        """
        last = CORE_END - 1
        change, cosine, ratio = [], [], []
        for cat, dump in self.dumps:
            if cat != category:
                continue
            mask = RoutingAnalysis._mask(cat, dump)
            a = dump.hidden(layer_pass_to_slot(last, first))[mask]
            b = dump.hidden(layer_pass_to_slot(last, second))[mask]
            norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            change += list(np.linalg.norm(b - a, axis=1) / norm_a)
            cosine += list(np.sum(a * b, axis=1) / (norm_a * norm_b))
            ratio += list(norm_b / norm_a)
        return {"relative_change": float(np.median(change)), "cosine": float(np.median(cosine)),
                "norm_ratio": float(np.median(ratio))}


def core_layers() -> range:
    """Return the physical layers inside the loop core."""
    return range(CORE_START, CORE_END)


def first_pass_slots() -> range:
    """Return the slots of the first visit of every layer."""
    return range(N_PHYS)
