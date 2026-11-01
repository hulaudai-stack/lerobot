"""Which posture should the other joints blend to while one joint sweeps to a stop?

For each end of `joint`, every target of `others` on a grid is scored by the worst clearance between
the arm and the table, clamps and wall (and no contact at all, the arm's own links included) while
the joint sweeps from a sampled middle-pose start to its stop and the others turn linearly from
where they started to the target. Usage:

    python posture_search.py shoulder_lift elbow_flex,wrist_flex 0.035

The shoulder's `Toward` posture in LeRobot comes from this search.
"""

import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101sim import JOINTS, build_model  # noqa: E402

ENV = ("table", "wall")


class Arm:
    def __init__(self, gap=0.035):
        self.m = build_model(wall_gap=gap, clamps=True)
        self.d = mujoco.MjData(self.m)
        self.adr = {j: self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINTS}
        self.rng = {j: self.m.jnt_range[self.m.joint(j).id].copy() for j in JOINTS}
        envb = {0}
        self.env = [
            g
            for g in range(self.m.ngeom)
            if self.m.geom_bodyid[g] in envb
            or (self.m.geom(g).name or "").startswith(("table", "wall", "clamp"))
        ]
        base = self.m.body("base").id if "base" in [self.m.body(i).name for i in range(self.m.nbody)] else 1
        self.arm = [
            g
            for g in range(self.m.ngeom)
            if self.m.geom_bodyid[g] not in (0, base) and self.m.geom_contype[g] | self.m.geom_conaffinity[g]
        ]
        self.fromto = np.zeros(6)

    def set(self, pose):
        self.d.qpos[:] = 0
        for j, a in pose.items():
            self.d.qpos[self.adr[j]] = a
        mujoco.mj_forward(self.m, self.d)

    def touching(self):
        return self.d.ncon > 0

    def clearance(self, cap=0.1):
        best = cap
        for g in self.arm:
            for e in self.env:
                best = min(best, mujoco.mj_geomDistance(self.m, self.d, g, e, cap, self.fromto))
        return best


def sweep(arm, joint, end, pose, n=16):
    """Min clearance sweeping `joint` from 0 to its stop at `end`; -1 if anything touches."""
    stop = arm.rng[joint][0 if end < 0 else 1]
    worst = 1.0
    for a in np.linspace(0, stop, n):
        arm.set({**pose, joint: a})
        if arm.touching():
            return -1.0
    for a in np.linspace(0, stop, 6):
        arm.set({**pose, joint: a})
        worst = min(worst, arm.clearance())
    return worst


joint = sys.argv[1]
others = sys.argv[2].split(",")
gap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.035
arm = Arm(gap)
rng = np.random.default_rng(0)
SPREAD = {"shoulder_pan": 30, "shoulder_lift": 20, "elbow_flex": 20, "wrist_flex": 20, "wrist_roll": 60}
starts = []
while len(starts) < 24:
    p = {j: np.radians(rng.uniform(-s, s)) for j, s in SPREAD.items()}
    p["gripper"] = np.radians(rng.uniform(0, 60))
    arm.set(p)
    if not arm.touching() and arm.d.xpos[arm.m.body("gripper").id][2] > 0.04:
        starts.append(p)


def score(end, target, starts, n=10):
    stop = arm.rng[joint][0 if end < 0 else 1]
    worst = 1.0
    for st in starts:
        for t in np.linspace(0, 1, n)[1:]:
            pose = dict(st)
            pose[joint] = st[joint] + t * (stop - st[joint])
            for j, v in target.items():
                pose[j] = st[j] + t * (v - st[j])
            arm.set(pose)
            if arm.touching():
                return -1.0
            if t > 0.5:
                worst = min(worst, arm.clearance())
    return worst


N = 13
grids = [np.linspace(*arm.rng[j], N) for j in others]
for end in (-1, 1):
    held = score(end, {}, starts)
    res = []
    for combo in itertools.product(*grids):
        target = dict(zip(others, combo, strict=True))
        s = score(end, target, starts[:6])  # screen on a few starts, then all
        if s >= 0:
            res.append((s, combo))
    res.sort(key=lambda r: -r[0])
    final = sorted(
        ((score(end, dict(zip(others, c, strict=True)), starts), c) for _, c in res[:12]), key=lambda r: -r[0]
    )
    print(
        f"{joint} end {end:+d} gap {gap}: held-at-start clearance {held * 100:.1f} cm; screened clean {len(res)}"
    )
    for s, c in final[:4]:
        print(
            f"   {s * 100:5.1f} cm  "
            + " ".join(f"{j}={np.degrees(v):+.0f}" for j, v in zip(others, c, strict=True))
        )
