"""Which mechanical stops of the SO-101 can be reached without touching anything?

For each joint and each end of its range, sample many poses of the other joints with that joint
at its stop, and keep the poses without any contact (self, table, wall, clamps). A stop that no
pose reaches cleanly cannot be found by any calibration that avoids pressing on the environment;
for those, the furthest clean angle is reported.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101sim import JOINTS, build_model  # noqa: E402


def clean(model, data, qpos) -> bool:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data.ncon == 0


def reach(model, joint: str, end: int, samples: int, rng) -> tuple[float, float, np.ndarray | None]:
    """The furthest angle towards `end` (-1 low, +1 high) of `joint` with a clean pose, the stop
    itself, and a clean pose at that angle."""
    data = mujoco.MjData(model)
    jid = model.joint(joint).id
    adr = model.jnt_qposadr[jid]
    low, high = model.jnt_range[jid]
    stop = high if end > 0 else low
    ranges = model.jnt_range[: model.njnt]
    others = rng.uniform(ranges[:, 0], ranges[:, 1], size=(samples, model.njnt))
    # Sweep from the stop inwards: the first angle with a clean pose is the furthest reachable.
    for angle in np.linspace(stop, 0.0, 61):
        for qpos in others:
            qpos = qpos.copy()
            qpos[adr] = angle
            if clean(model, data, qpos):
                return float(angle), float(stop), qpos
    return float("nan"), float(stop), None


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for label, gap in (("no environment", "none"), ("table, clamps, wall 2 cm", 0.02), ("wall 0 cm", 0.0)):
        model = build_model(wall_gap=None if gap == "none" else gap, clamps=gap != "none")
        if gap == "none":
            # Without the table either: remove it by making it not collide.
            model.geom_contype[model.geom("table").id] = 0
            model.geom_conaffinity[model.geom("table").id] = 0
        print(f"== {label}")
        for joint in JOINTS:
            if joint == "wrist_roll":
                continue
            row = []
            for end in (-1, 1):
                angle, stop, _ = reach(model, joint, end, 3000, rng)
                row.append(
                    f"{'low' if end < 0 else 'high'} stop {np.degrees(stop):7.1f}°: reachable to {np.degrees(angle):7.1f}°"
                )
            print(f"  {joint:14s} " + " | ".join(row))
