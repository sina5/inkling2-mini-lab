"""Build every chart and a summary JSON from the files in runs/. Skip a chart when its runs are missing."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inklab.analysis import LoopAnalysis, RoutingAnalysis, RunSummarizer, confusion_matrix, core_layers
from inklab.constants import CIFAR_CLASSES, FIGURES_DIR, REASONING_EFFORTS, RUNS_DIR
from inklab.probe_data import CORE_END, CORE_START, N_PHYS, ProbeReader

DPI = 200
SIZE = (16, 9)
COLORS = ["#2364aa", "#e4572e", "#29a36a", "#f3a712", "#8e44ad", "#17becf"]
SERIES_RESULTS = [  # CIFAR-10 test accuracy from the earlier posts of the series (full 10,000-image test set)
    ("ResNet-18, trained", 94.9),
    ("Small ViT, trained", 92.9),
    ("Joint diffusion model", 87.2),
    ("Diffusion classifier", 86.7),
]
CATEGORIES = ("image", "english", "code", "math", "other_language")
LABELS = {"image": "Image tokens", "english": "English", "code": "Code", "math": "Math", "other_language": "Other languages"}


def execution_order() -> list[int]:
    """Return cache slots in the order the model runs them: prelude, three core passes, coda."""
    return (list(range(CORE_START)) + list(range(CORE_START, CORE_END))
            + list(range(N_PHYS, N_PHYS + 2 * (CORE_END - CORE_START))) + list(range(CORE_END, N_PHYS)))


class FigureMaker:
    """Write charts to FIGURES_DIR and collect numbers for the summary."""

    def __init__(self) -> None:
        self.summary: dict = {}
        self.summarizer = RunSummarizer()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.size": 17, "axes.titlesize": 22, "axes.labelsize": 18, "figure.facecolor": "white"})

    def run(self) -> None:
        """Make every chart whose data exists, then write summary.json."""
        for step in (self.cifar_compare, self.confusion, self.loop_passes, self.effort, self.speed, self.probes):
            step()
        (RUNS_DIR / "summary.json").write_text(json.dumps(self.summary, indent=2), encoding="utf-8")

    def _save(self, fig: plt.Figure, name: str) -> None:
        fig.savefig(FIGURES_DIR / name, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {name}")

    def _cifar(self, name: str):
        path = RUNS_DIR / f"{name}.jsonl"
        return self.summarizer.cifar(path) if path.exists() else None

    def _gsm8k(self, name: str):
        path = RUNS_DIR / f"{name}.jsonl"
        return self.summarizer.gsm8k(path) if path.exists() else None

    def cifar_compare(self) -> None:
        """Bar chart: zero-shot Inkling 2 Mini next to the trained models of the series."""
        runs = [("Inkling 2 Mini, zero-shot, 240 px", self._cifar("cifar_main_pdefault_none_s240")),
                ("Inkling 2 Mini, zero-shot, 32 px", self._cifar("cifar_main_pdefault_none_s32"))]
        runs = [(label, s) for label, s in runs if s is not None]
        if not runs:
            return
        self.summary["cifar"] = {label: s.__dict__ for label, s in runs}
        bars = [(label, 100 * s.accuracy, 100 * s.stderr, COLORS[1]) for label, s in runs]
        bars += [(label, value, 0.0, COLORS[0]) for label, value in SERIES_RESULTS]
        bars.sort(key=lambda b: b[1])
        fig, ax = plt.subplots(figsize=SIZE)
        y = np.arange(len(bars))
        ax.barh(y, [b[1] for b in bars], color=[b[3] for b in bars])
        for i, b in enumerate(bars):
            if b[2] > 0:
                ax.errorbar(b[1], i, xerr=b[2], color="black", capsize=6, lw=2)
            ax.text(b[1] + b[2] + 0.4, i, f"{b[1]:.1f}%", va="center", fontsize=17)
        ax.set_yticks(y, [b[0] for b in bars])
        ax.set_xlim(min(b[1] for b in bars) - 15, 100)
        ax.set_xlabel("CIFAR-10 test accuracy (%)")
        n = runs[0][1].n
        ax.set_title(f"No CIFAR-10 training vs. trained models\nOrange: Inkling 2 Mini on {n:,} balanced test images (±1 s.e.). Blue: earlier posts, full test set",
                     loc="left")
        ax.grid(axis="x", alpha=0.3)
        self._save(fig, "cifar_compare.png")

    def confusion(self) -> None:
        """Confusion matrix of the main 240 px run, in percent of each true class."""
        path = RUNS_DIR / "cifar_main_pdefault_none_s240.jsonl"
        if not path.exists():
            return
        counts = confusion_matrix(path, len(CIFAR_CLASSES))
        share = 100 * counts / counts.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=SIZE)
        image = ax.imshow(share, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(share.shape[0]):
            for j in range(share.shape[1]):
                if counts[i, j]:
                    ax.text(j, i, f"{share[i, j]:.0f}", ha="center", va="center", fontsize=14,
                            color="white" if share[i, j] > 60 else "black")
        ax.set_xticks(range(len(CIFAR_CLASSES) + 1), list(CIFAR_CLASSES) + ["no class"], rotation=30, ha="right")
        ax.set_yticks(range(len(CIFAR_CLASSES)), CIFAR_CLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"Where zero-shot Inkling 2 Mini goes wrong (% of each true class, {counts.sum():,} images)", loc="left")
        fig.colorbar(image, ax=ax, fraction=0.03)
        self.summary["confusion_240"] = counts.tolist()
        self._save(fig, "cifar_confusion.png")

    def loop_passes(self) -> None:
        """Accuracy and speed of the variable-loop checkpoint at 1, 2, and 3 passes."""
        cifar = {p: self._cifar(f"cifar_varloop_p{p}_none_s240") for p in (1, 2, 3)}
        gsm = {p: self._gsm8k(f"gsm8k_varloop_p{p}_low_n50") for p in (1, 2, 3)}
        cifar = {p: s for p, s in cifar.items() if s is not None}
        gsm = {p: s for p, s in gsm.items() if s is not None}
        if not cifar and not gsm:
            return
        self.summary["loop_passes"] = {"cifar": {p: s.__dict__ for p, s in cifar.items()},
                                       "gsm8k": {p: s.__dict__ for p, s in gsm.items()}}
        fig, axes = plt.subplots(1, 2, figsize=SIZE)
        for ax, data, title in ((axes[0], cifar, "CIFAR-10, zero-shot"), (axes[1], gsm, "GSM8K, effort low")):
            passes = sorted(data)
            acc = [100 * data[p].accuracy for p in passes]
            err = [100 * data[p].stderr for p in passes]
            ax.errorbar(passes, acc, yerr=err, marker="o", markersize=12, lw=3, capsize=8, color=COLORS[0])
            for p, a, s in zip(passes, acc, (data[p] for p in passes)):
                ax.annotate(f"{a:.1f}%\n{s.mean_tokens:.0f} tok\n{s.mean_seconds:.1f} s/item", (p, a),
                            textcoords="offset points", xytext=(14, -48), ha="left", fontsize=14,
                            bbox={"boxstyle": "round", "fc": "white", "ec": "lightgray"})
            ax.set_xticks([1, 2, 3])
            ax.set_xlabel("Passes through the loop core (layers 6-17)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xlim(0.6, 3.8)
            if acc:
                ax.set_ylim(min(acc) - 12, max(a + e for a, e in zip(acc, err)) + 2)
            n = data[passes[0]].n if passes else 0
            ax.set_title(f"{title} ({n} items)", loc="left")
            ax.grid(alpha=0.3)
        fig.suptitle("Variable-loop research checkpoint: same weights, different loop depth", x=0.01, ha="left", fontsize=22)
        fig.tight_layout()
        self._save(fig, "loop_passes.png")

    def effort(self) -> None:
        """GSM8K accuracy against generated tokens for each reasoning effort."""
        data = [(e, self._gsm8k(f"gsm8k_main_pdefault_{e}_n50")) for e in REASONING_EFFORTS]
        data = [(e, s) for e, s in data if s is not None]
        if not data:
            return
        self.summary["effort"] = {e: s.__dict__ for e, s in data}
        fig, ax = plt.subplots(figsize=SIZE)
        x = [s.mean_tokens for _, s in data]
        y = [100 * s.accuracy for _, s in data]
        ax.errorbar(x, y, yerr=[100 * s.stderr for _, s in data], fmt="o", markersize=14, lw=3, capsize=8,
                    color=COLORS[1])
        for i, ((e, s), xi, yi) in enumerate(zip(data, x, y)):
            ax.annotate(f"{e}: {yi:.0f}%\n{s.mean_tokens:.0f} tok, {s.mean_seconds:.0f} s", (xi, yi),
                        textcoords="offset points", xytext=(16, -60 if i % 2 else 22), ha="left", fontsize=15,
                        bbox={"boxstyle": "round", "fc": "white", "ec": "lightgray"})
        ax.set_xlabel("Mean generated tokens per question, reasoning included")
        ax.set_ylabel("GSM8K accuracy (%)")
        ax.set_xlim(min(x) * 0.85, max(x) * 1.2)
        ax.set_ylim(min(y) - 15, 104)
        ax.set_title(f"Reasoning effort: accuracy vs. tokens spent (first {data[0][1].n} GSM8K test questions)", loc="left")
        ax.grid(alpha=0.3, which="both")
        self._save(fig, "effort.png")

    def speed(self) -> None:
        """Prompt and generation speed for each expert offload level."""
        path = RUNS_DIR / "bench_speed.json"
        if not path.exists():
            return
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.summary["speed"] = [{k: r[k] for k in ("n_cpu_moe", "n_prompt", "n_gen", "avg_ts", "stddev_ts")} for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=SIZE)
        for ax, kind, color in ((axes[0], "n_prompt", COLORS[0]), (axes[1], "n_gen", COLORS[2])):
            sel = sorted((r for r in rows if r[kind]), key=lambda r: r["n_cpu_moe"])
            xs = [r["n_cpu_moe"] for r in sel]
            ys = [r["avg_ts"] for r in sel]
            ax.errorbar(xs, ys, yerr=[r["stddev_ts"] for r in sel], marker="o", markersize=11, lw=3, capsize=6, color=color)
            for xi, yi in zip(xs, ys):
                ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=14)
            ax.set_xlabel("Layers with expert weights in CPU RAM (--n-cpu-moe)")
            ax.set_ylabel("Tokens per second")
            ax.set_ylim(0, max(ys) * 1.2)
            ax.set_title("Prompt processing (512 tokens)" if kind == "n_prompt" else "Generation (128 tokens)", loc="left")
            ax.grid(alpha=0.3)
        fig.suptitle("Inkling 2 Mini IQ4_NL on one RTX 3080 Ti (12 GB), micro-batch 16", x=0.01, ha="left", fontsize=22)
        fig.tight_layout()
        self._save(fig, "speed.png")

    def probes(self) -> None:
        """Routing and loop-gate charts from the probe dumps."""
        index_path = RUNS_DIR / "probe" / "index.json"
        if not index_path.exists():
            return
        index = json.loads(index_path.read_text(encoding="utf-8"))
        dumps = [(e["category"], ProbeReader(RUNS_DIR / "probe" / f"{e['name']}.bin").read()) for e in index]
        routing = RoutingAnalysis(dumps)
        loop = LoopAnalysis(dumps)
        self._routing_divergence(routing)
        self._routing_heatmap(routing)
        self._loop_gates(loop, routing)

    def _routing_divergence(self, routing: RoutingAnalysis) -> None:
        order = execution_order()
        fig, ax = plt.subplots(figsize=SIZE)
        result = {}
        for color, cat in zip(COLORS[1:], ("image", "code", "math", "other_language")):
            values = [routing.divergence(cat, "english", slot) for slot in order]
            result[cat] = values
            ax.plot(range(len(order)), values, lw=3, marker="o", markersize=5, color=color, label=f"{LABELS[cat]} vs. English")
        bounds = [CORE_START, CORE_END, CORE_END + 12, CORE_END + 24]
        for b, text in zip(bounds, ("pass 1", "pass 2", "pass 3", "coda")):
            ax.axvline(b - 0.5, color="gray", ls="--", lw=1.5)
            ax.text(b, ax.get_ylim()[1] * 0.97, text, fontsize=15, va="top", color="dimgray")
        ax.text(0, ax.get_ylim()[1] * 0.97, "prelude", fontsize=15, va="top", color="dimgray")
        ax.set_xticks(range(0, len(order), 6), [str(i) for i in range(0, len(order), 6)])
        ax.set_xlabel("Layer visit in execution order (48 visits per token)")
        ax.set_ylabel("Jensen-Shannon divergence of expert use (bits)")
        ax.set_title("How differently the router treats each kind of input, compared with English text", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
        ax.grid(alpha=0.3)
        self.summary["routing_divergence"] = result
        self._save(fig, "routing_divergence.png")

    def _routing_heatmap(self, routing: RoutingAnalysis) -> None:
        layer = 12
        usage = np.stack([routing.usage(cat, layer) for cat in CATEGORIES])
        order = np.argsort(-usage[0])
        fig, ax = plt.subplots(figsize=SIZE)
        image = ax.imshow(100 * usage[:, order], aspect="auto", cmap="magma_r")
        ax.set_yticks(range(len(CATEGORIES)), [LABELS[c] for c in CATEGORIES])
        ax.set_xlabel("Expert (sorted by how often image tokens use it)")
        ax.set_title(f"Share of routing picks per expert, layer {layer}, first pass", loc="left")
        fig.colorbar(image, ax=ax, fraction=0.03, label="% of picks")
        top = usage[:, order[:8]].sum(axis=1)
        self.summary["heatmap_top8_share_of_image_experts"] = dict(zip(CATEGORIES, top.tolist()))
        self._save(fig, "routing_heatmap.png")

    def _loop_gates(self, loop: LoopAnalysis, routing: RoutingAnalysis) -> None:
        layers = list(core_layers())
        fig, axes = plt.subplots(1, 2, figsize=SIZE)
        gates = {}
        for ax, kind in ((axes[0], "gain"), (axes[1], "forget")):
            for color, cat in zip(COLORS, ("image", "english", "code")):
                for pass_number, style in ((2, "-"), (3, ":")):
                    values = [loop.gate_strength(kind, "ffn", layer, pass_number, cat) for layer in layers]
                    gates[f"{kind}_{cat}_p{pass_number}"] = values
                    ax.plot(layers, values, ls=style, lw=3, marker="o", color=color,
                            label=f"{LABELS[cat]}, pass {pass_number}")
            ax.set_xlabel("Core layer")
            ax.set_ylabel(f"Mean |{kind}| (tanh output, 0 to 1)")
            ax.set_title(f"{kind.capitalize()} gate on the FFN input", loc="left")
            ax.grid(alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle("How much each loop pass takes from the previous pass (gain) and drops of its own input (forget)",
                     x=0.01, ha="left", fontsize=20)
        fig.tight_layout()
        self.summary["gates"] = gates
        self.summary["pass_change"] = {cat: {f"{a}-{b}": loop.pass_change(a, b, cat) for a, b in ((1, 2), (2, 3), (1, 3))}
                                       for cat in CATEGORIES}
        self.summary["probe_token_counts"] = {cat: int(sum(RoutingAnalysis._mask(c, d).sum() for c, d in loop.dumps if c == cat))
                                              for cat in CATEGORIES}
        self.summary["pass_overlap"] = {cat: {f"{a}-{b}": float(np.mean([routing.pass_overlap(layer, a, b, cat)
                                                                           for layer in layers]))
                                              for a, b in ((1, 2), (2, 3), (1, 3))} for cat in CATEGORIES}
        self._save(fig, "loop_gates.png")


if __name__ == "__main__":
    FigureMaker().run()
