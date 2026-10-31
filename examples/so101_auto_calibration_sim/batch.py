"""Run the simulated calibration over many random assemblies, start poses and wall gaps.

Usage: python batch.py OUT_DIR SEED_FROM SEED_TO [WALL_GAP ...]
Each run writes OUT_DIR/seed{N}_gap{G}.json; a summary table is printed at the end.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REACHABLE = {"elbow_flex": (-96.8, 90.4)}
TOLERANCE_DEG = 3.0


def one(args):
    seed, gap, out = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_calibration import run

    path = Path(out) / f"seed{seed}_gap{gap}.json"
    summary, lines, _ = run(seed, gap, None, 10.0, 1.0)
    path.write_text(json.dumps(summary))
    (path.with_suffix(".log")).write_text("\n".join(lines))
    return summary


def verdict(summary):
    if summary["error"]:
        return False, summary["error"][:80]
    bad = []
    for joint, result in summary["results"].items():
        low, high = REACHABLE.get(joint, result["true"])
        short_low, short_high = result["found"][0] - low, high - result["found"][1]
        if short_low > TOLERANCE_DEG or short_high > TOLERANCE_DEG:
            bad.append(f"{joint} short {short_low:+.1f}/{short_high:+.1f}")
    return not bad, "; ".join(bad) or "ok"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    seeds = range(int(sys.argv[2]), int(sys.argv[3]) + 1)
    gaps = [float(g) for g in sys.argv[4:]] or [0.04]
    jobs = [(seed, gap, str(out)) for gap in gaps for seed in seeds]
    passed = 0
    with ProcessPoolExecutor(max_workers=2) as pool:
        for summary in pool.map(one, jobs):
            ok, why = verdict(summary)
            passed += ok
            print(
                f"seed {summary['seed']:3d} gap {summary['wall_gap']}: {'PASS' if ok else 'FAIL'}  {why}",
                flush=True,
            )
    print(f"{passed}/{len(jobs)} passed")
