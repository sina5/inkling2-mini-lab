"""Draw the Inkling 2 Mini layer schedule: prelude, three passes of the loop core, and coda."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from inklab.constants import FIGURES_DIR

BLUE, ORANGE, GREEN, GRAY = "#2364aa", "#e4572e", "#29a36a", "#555555"


class ArchitectureDiagram:
    """A left-to-right diagram of the 48 layer visits per token."""

    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(16, 9))
        self.ax.set_xlim(0, 16)
        self.ax.set_ylim(0, 9)
        self.ax.axis("off")

    def box(self, x: float, y: float, w: float, h: float, color: str, title: str, body: str) -> None:
        """Draw one rounded block with a title and a smaller body text."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                         fc=color, ec="black", lw=1.5, alpha=0.9))
        self.ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", color="white", fontsize=17, weight="bold")
        self.ax.text(x + w / 2, y + h * 0.3, body, ha="center", va="center", color="white", fontsize=13)

    def arrow(self, start: tuple[float, float], end: tuple[float, float], color: str = "black", style: str = "-|>",
              rad: float = 0.0, ls: str = "-") -> None:
        """Draw an arrow between two points."""
        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=22, lw=2.2, color=color,
                                          connectionstyle=f"arc3,rad={rad}", linestyle=ls))

    def draw(self) -> None:
        """Draw every block, arrow, and note."""
        ax = self.ax
        ax.text(0.2, 8.6, "Inkling 2 Mini: 24 layers of weights, 48 layer visits per token", fontsize=22, weight="bold")
        ax.text(0.2, 8.15, "Each layer: attention (5 of 6 use a 1,024-token sliding window) + a mixture of experts (8 of 128 picked per token)",
                fontsize=14, color=GRAY)
        self.box(0.2, 4.6, 1.8, 1.4, GRAY, "Input", "text tokens\n+ image tokens")
        self.box(2.6, 4.6, 2.2, 1.4, BLUE, "Prelude", "layers 0-5\nrun once")
        xs = (5.6, 8.3, 11.0)
        for i, x in enumerate(xs):
            self.box(x, 4.6, 2.2, 1.4, ORANGE, f"Core, pass {i + 1}", "layers 6-17\nsame weights")
        self.box(13.7, 4.6, 2.1, 1.4, BLUE, "Coda", "layers 18-23\nrun once")
        self.arrow((2.0, 5.3), (2.6, 5.3))
        self.arrow((4.8, 5.3), (5.6, 5.3))
        for x in xs[1:]:
            self.arrow((3.7, 6.05), (x + 1.1, 6.05), rad=-0.2)
        ax.text(8.6, 7.45, "every pass starts from the prelude output", ha="center", fontsize=15)
        self.arrow((13.2, 5.3), (13.7, 5.3))
        for a, b in ((xs[0], xs[1]), (xs[1], xs[2])):
            self.arrow((a + 1.1, 4.55), (b + 1.1, 4.55), color=GREEN, rad=0.45, ls="--")
        ax.text(9.4, 3.45, "message from the previous pass", ha="center", fontsize=15, color=GREEN)
        notes = [
            "Passes 2 and 3 start again from the prelude output, not from the end of the previous pass.",
            "Each core layer in passes 2 and 3 reads a 'message': an attention-weighted mix of the previous pass's block outputs.",
            "Small gates decide how much of the message to add (gain) and how much of the current input to drop (forget).",
            "Every visit keeps its own KV cache and convolution state, so 3 passes cost about 3x the core compute but no extra weights.",
            "26.4B parameters in total; about 3.4B are active per token, or 5.4B counting the repeated passes.",
        ]
        for i, note in enumerate(notes):
            ax.text(0.3, 2.7 - i * 0.55, f"• {note}", fontsize=14)

    def save(self) -> None:
        """Write figures/architecture.png."""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(FIGURES_DIR / "architecture.png", dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    diagram = ArchitectureDiagram()
    diagram.draw()
    diagram.save()
