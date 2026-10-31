"""Chart found vs true joint ranges for one or more simulated calibration runs (JSON summaries)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# True reachable stops: the model's joint ranges, except where the arm meets itself first.
REACHABLE = {"elbow_flex": (-96.8, 90.4)}


def chart(summaries: list[dict], path: Path, title: str) -> None:
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper"]
    fig, axes = plt.subplots(len(joints), 1, figsize=(8, 1.15 * len(joints) + 1), sharex=False)
    for ax, joint in zip(axes, joints, strict=True):
        for row, summary in enumerate(summaries):
            result = summary["results"].get(joint)
            label = f"seed {summary['seed']}" + (
                f" gap {summary['wall_gap']}" if summary.get("wall_gap") is not None else " no wall"
            )
            if result is None:
                ax.text(0, row, f"{label}: failed", va="center", ha="center", fontsize=8, color="#A32D2D")
                continue
            true_low, true_high = REACHABLE.get(joint, result["true"])
            low, high = result["found"]
            ax.plot([true_low, true_high], [row, row], color="#B4B2A9", lw=8, solid_capstyle="butt")
            ok = abs(low - true_low) <= 3 and abs(true_high - high) <= 3
            ax.plot(
                [low, high], [row, row], color="#1D9E75" if ok else "#D85A30", lw=3, solid_capstyle="butt"
            )
            ax.text(
                true_high + 4,
                row,
                f"{label}: short {low - true_low:+.1f}° / {true_high - high:+.1f}°",
                va="center",
                fontsize=7,
            )
        ax.set_yticks([])
        ax.set_ylim(-0.7, len(summaries) - 0.3)
        ax.set_title(joint, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.set_xlim(-125, 200)
    fig.suptitle(title + "\ngray = true stops, green = found within 3°, orange = short", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    summaries = [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
    chart(summaries, out, "Simulated auto-calibration: found vs true joint ranges")
