"""Test the SO-101 automatic calibration from many starting points, then tune its settings.

Runs unattended and can be resumed: everything is saved under OUT_DIR as it goes.

    python optimize.py OUT_DIR [--scenarios 24] [--candidates 24] [--workers 2]

Phase 1, robustness: the current settings against scenarios that each draw a servo assembly
(mounting angle and direction of every encoder), a start pose, and a wall 3.5 cm to 20 cm behind
the base, or none. The start pose is roughly the middle pose, as a person places the arm (within
20 degrees of it per joint, 30 for the pan), or with START_SPREAD=stress, anywhere within ~45% of
each joint's range.

Phase 2, optimisation: random settings (ramp speed, torque ladder, stall detection, survey size,
postures searched) by successive halving. A candidate that misses a single stop by more than
3 degrees in any scenario is out; the others are ranked by calibration time, and the fastest are
tried on more scenarios. Results go to OUT_DIR/leaderboard.md as they come.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

if sys.platform.startswith("linux"):  # software rendering for headless Linux; macOS refuses it
    os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101sim  # noqa: E402,F401  (puts the LeRobot checkout under test first on the path)

REACHABLE = {"elbow_flex": (-96.8, 90.4)}
TOLERANCE_DEG = 3.0
WALL_GAPS = [0.035, 0.04, 0.06, 0.10, 0.20, None]

# Postures each candidate may search from, by name. "default" is what LeRobot ships.
POSTURE_SETS = {
    "default": None,
    "lean_shoulder": {"shoulder_lift": "follow_only"},
    "lean_elbow": {"elbow_flex": "wrist_bent_only"},
    "lean_both": {"shoulder_lift": "follow_only", "elbow_flex": "wrist_bent_only"},
}

DEFAULTS = {
    "step": 8,
    "move_step": 8,
    "torque_limits": [150, 300, 500],
    "stall_error": 150,
    "stall_cycles": 3,
    "survey_ticks": 100,
    "postures": "default",
    "load_stall": 0.9,
    "trust_impacts": True,
    "working_torque_limit": None,
}
# Speeds stay at 8 ticks per 20 ms, and the first push at 15%: faster, or harder, wedges a hand
# that starts near the table (see README). The rest is tuned for force first, then time.
SPACE = {
    "step": [8],
    "move_step": [8],
    "torque_limits": [[150, 300], [150, 250, 350], [150, 300, 400], [150, 300, 500], [150, 250]],
    "stall_error": [150, 200],
    "stall_cycles": [3, 4],
    "survey_ticks": [100],
    "postures": list(POSTURE_SETS),
    "load_stall": [0.8, 0.9],
    "trust_impacts": [True],
    "working_torque_limit": [300, 350, 400, 500, None],
}


def postures_for(name: str):
    from lerobot.motors import Follow
    from lerobot.robots.so_follower.so_follower import _HARD_STOP_SEARCH_POSTURES

    postures = {joint: list(options) for joint, options in _HARD_STOP_SEARCH_POSTURES.items()}
    lean = POSTURE_SETS[name] or {}
    if lean.get("shoulder_lift") == "follow_only":
        postures["shoulder_lift"] = [
            p for p in postures["shoulder_lift"] if any(isinstance(v, Follow) for v in p.values())
        ]
    if lean.get("elbow_flex") == "wrist_bent_only":
        postures["elbow_flex"] = [p for p in postures["elbow_flex"] if p]
    assert all(postures.values())  # every joint keeps at least one posture
    return postures


# How far from the middle pose a scenario's start may be, in degrees per joint: "roughly the middle
# pose", as a person places the arm following the instructions, or a stress test far beyond that.
START_SPREADS = {
    "middle": {"shoulder_pan": 30, "shoulder_lift": 20, "elbow_flex": 20, "wrist_flex": 20, "wrist_roll": 60},
    "stress": {"shoulder_pan": 50, "shoulder_lift": 45, "elbow_flex": 43, "wrist_flex": 43, "wrist_roll": 70},
}
SPREAD = os.environ.get("START_SPREAD", "middle")


def scenario(index: int) -> dict:
    """A servo assembly, a wall and a start pose that touches nothing, all drawn from `index`."""
    import mujoco
    import numpy as np
    from so101sim import JOINTS, SimArm, build_model, random_servos

    rng = np.random.default_rng(10_000 + index)
    gap = WALL_GAPS[index % len(WALL_GAPS)]
    model = build_model(wall_gap=gap)
    servos = random_servos(rng)
    arm = SimArm(model, servos, record=False)
    gripper_body = model.body("gripper").id
    for _ in range(1000):
        pose = {}
        for joint in JOINTS:
            if joint == "gripper":
                continue
            spread = START_SPREADS[SPREAD][joint]
            pose[joint] = float(np.radians(rng.uniform(-spread, spread)))
        pose["gripper"] = float(np.radians(rng.uniform(0, 60)))
        for joint, angle in pose.items():
            arm.data.qpos[arm.qadr[joint]] = angle
        mujoco.mj_forward(model, arm.data)
        if arm.data.ncon == 0 and arm.data.xpos[gripper_body][2] > 0.04:
            break
    return {
        "index": index,
        "wall_gap": gap,
        "servos": {j: [s.mount, s.direction] for j, s in servos.items()},
        "pose": pose,
    }


def run_case(job: tuple[dict, dict], video: Path | None = None) -> dict:
    params, case = job
    import mujoco
    import numpy as np
    from so101sim import JOINTS, Servo, SimArm, SimBus, build_model

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    logging.getLogger("lerobot").setLevel(logging.WARNING)
    model = build_model(wall_gap=case["wall_gap"])
    arm = SimArm(
        model,
        {j: Servo(m, d) for j, (m, d) in case["servos"].items()},
        record=video is not None,
        frame_every_s=0.25,
    )
    arm.trace_every_s = 1e9
    arm.noise_ticks = int(os.environ.get("NOISE_TICKS", "0"))
    for joint, angle in case["pose"].items():
        arm.data.qpos[arm.qadr[joint]] = angle
    mujoco.mj_forward(model, arm.data)

    config = SO101FollowerConfig(
        port="/dev/sim",
        id=f"opt{case['index']}",
        calibration_dir=Path(tempfile.mkdtemp()),
        auto_calibrate=True,
    )
    config.auto_calibration_torque_limits = list(params["torque_limits"])
    robot = SO101Follower(config)
    robot.bus._bus = SimBus(arm, robot.bus._definitions)
    original = robot.bus.record_calibration_from_hard_stops

    def tuned(*args, **kwargs):
        kwargs.update(
            step=params["step"],
            move_step=params.get("move_step", params["step"]),
            stall_error=params["stall_error"],
            **{
                key: params[key]
                for key in ("load_stall", "trust_impacts", "working_torque_limit")
                if key in params
            },
            stall_cycles=params["stall_cycles"],
            survey_ticks=params["survey_ticks"],
            postures=postures_for(params["postures"]),
        )
        return original(*args, **kwargs)

    robot.bus.record_calibration_from_hard_stops = tuned
    handler = None
    if video is not None:
        from so101sim import CaptionHandler

        handler = CaptionHandler(arm)
        logging.getLogger("lerobot.motors.motors_bus").addHandler(handler)
        logging.getLogger("lerobot.motors.motors_bus").setLevel(logging.INFO)
    error, started = None, time.time()
    with (
        patch("lerobot.motors.motors_bus.time.sleep", arm.sleep),
        patch("builtins.input", lambda prompt="": ""),
        patch("builtins.print", lambda *a, **k: None),
    ):
        try:
            robot.calibrate()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]

    if video is not None:
        from so101sim import write_video

        write_video(arm.frames, video)
        video.with_suffix(".log").write_text("\n".join(handler.lines))
    worst, shortfalls = 0.0, {}
    if error is None:
        for joint in JOINTS:
            if joint == "wrist_roll":
                continue
            cal = robot.calibration[joint]
            true_low, true_high = REACHABLE.get(joint, np.degrees(model.jnt_range[model.joint(joint).id]))
            ends = sorted(
                np.degrees(arm.angle_of_reading(joint, reading, cal.homing_offset))
                for reading in (cal.range_min - 20, cal.range_max + 20)
            )
            short = (round(ends[0] - true_low, 2), round(true_high - ends[1], 2))
            shortfalls[joint] = short
            worst = float(max(worst, *short))
    from so101sim import MAX_TORQUE_NM

    stressed = [j for j in JOINTS if j != "wrist_roll"]
    return {
        "case": case["index"],
        "wall_gap": case["wall_gap"],
        "peak_pct": round(max(arm.peak_torque[j] for j in stressed) / MAX_TORQUE_NM * 100, 1),
        "hard_push_s": round(sum(arm.hard_push_s[j] for j in stressed), 2),
        "stress": {
            j: [round(arm.peak_torque[j] / MAX_TORQUE_NM * 100, 1), round(arm.hard_push_s[j], 2)]
            for j in stressed
        },
        "error": error,
        "passed": bool(error is None and worst <= TOLERANCE_DEG),
        "worst_short_deg": round(worst, 2) if error is None else None,
        "shortfalls": shortfalls,
        "sim_s": round(arm.time_s, 1),
        "wall_s": round(time.time() - started, 1),
    }


def stress_score(results: list[dict]) -> float:
    """What the optimiser minimises among settings that find every stop: the mean peak torque, in
    % of the servos' maximum, plus the mean seconds spent above 20% of it, plus a twentieth of the
    mean calibration time in seconds."""
    n = len(results)
    return (
        sum(r["peak_pct"] for r in results) / n
        + sum(r["hard_push_s"] for r in results) / n
        + 0.05 * sum(r["sim_s"] for r in results) / n
    )


def summary(results: list[dict]) -> str:
    n = len(results)
    return (
        f"{sum(r['passed'] for r in results)}/{n} pass, peak {sum(r['peak_pct'] for r in results) / n:.0f}% "
        f"(max {max(r['peak_pct'] for r in results):.0f}%), {sum(r['hard_push_s'] for r in results) / n:.1f}s "
        f"above 20%, {sum(r['sim_s'] for r in results) / n:.0f}s"
    )


class Study:
    def __init__(self, out: Path, workers: int):
        self.out, self.workers = out, workers
        self._pool: ProcessPoolExecutor | None = None
        self.state_path = out / "state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"runs": {}, "scenarios": {}}
        )

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state))

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.out / "progress.log", "a") as f:
            f.write(line + "\n")

    def case(self, index: int) -> dict:
        key = str(index)
        if key not in self.state["scenarios"]:
            self.state["scenarios"][key] = scenario(index)
            self.save()
        return self.state["scenarios"][key]

    def evaluate(self, name: str, params: dict, indices: list[int]) -> list[dict]:
        done = self.state["runs"].setdefault(name, {"params": params, "results": {}})["results"]
        todo = [i for i in indices if str(i) not in done]
        if todo:
            if self._pool is None:
                # One pool for the whole study: its workers import LeRobot once.
                self._pool = ProcessPoolExecutor(max_workers=self.workers)
            for result in self._pool.map(run_case, [(params, self.case(i)) for i in todo]):
                done[str(result["case"])] = result
                self.save()
                verdict = "PASS" if result["passed"] else "FAIL"
                why = result["error"] or f"worst short {result['worst_short_deg']}°"
                self.log(
                    f"{name}: case {result['case']} wall {result['wall_gap']} {verdict} {result['sim_s']}s sim ({why})"
                )
        results = [done[str(i)] for i in indices]
        self.leaderboard()
        return results

    def leaderboard(self) -> None:
        rows = []
        for name, run in self.state["runs"].items():
            results = list(run["results"].values())
            if not results:
                continue
            passed = sum(r["passed"] for r in results)
            n = len(results)
            rows.append(
                (
                    passed < n,
                    stress_score(results),
                    name,
                    run["params"],
                    passed,
                    n,
                    sum(r["peak_pct"] for r in results) / n,
                    sum(r["hard_push_s"] for r in results) / n,
                    sum(r["sim_s"] for r in results) / n,
                )
            )
        rows.sort()
        lines = [
            "# Auto-calibration settings leaderboard",
            "",
            "Pass = every stop found within 3°. Peak = largest torque any joint applied, % of the servo's"
            " maximum, mean over scenarios. Above 20% = seconds per calibration any joint spent above 20%."
            " Time = simulated seconds per calibration. Sorted by force, then time, among settings that pass.",
            "",
            "| Settings | Passed | Peak | Above 20% | Time (s) | torque ladder | working limit | stall lead "
            "| stall cycles | load stall | postures |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, _, name, p, passed, total, peak, hard, mean_s in rows:
            lines.append(
                f"| {name} | {passed}/{total} | {peak:.0f}% | {hard:.1f}s | {mean_s:.0f} | {p['torque_limits']} "
                f"| {p.get('working_torque_limit')} | {p['stall_error']} | {p['stall_cycles']} "
                f"| {p.get('load_stall')} | {p['postures']} |"
            )
        (self.out / "leaderboard.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--scenarios", type=int, default=24)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--replay", type=int, nargs="*", help="scenarios to replay on video, then exit")
    parser.add_argument(
        "--holdout", type=int, nargs=2, metavar=("FIRST", "LAST"), help="scenarios to check the best on"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    study = Study(args.out, args.workers)
    if args.replay is not None:
        for index in args.replay:
            result = run_case((DEFAULTS, study.case(index)), video=args.out / f"case{index}.mp4")
            print(json.dumps(result))
        return

    study.log(f"Phase 1: current settings on {args.scenarios} scenarios")
    baseline = study.evaluate("current", DEFAULTS, list(range(args.scenarios)))
    failed = [r for r in baseline if not r["passed"]]
    study.log(f"Phase 1 done: {len(baseline) - len(failed)}/{len(baseline)} passed")
    if failed:
        study.log(
            "Failures to fix before tuning: "
            + "; ".join(f"case {r['case']}: {r['error'] or r['shortfalls']}" for r in failed)
        )

    rng = random.Random(0)
    candidates, seen = {}, {json.dumps(DEFAULTS, sort_keys=True)}
    while len(candidates) < args.candidates:
        params = {key: rng.choice(values) for key, values in SPACE.items()}
        key = json.dumps(params, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates[f"c{len(candidates):02d}"] = params

    # Successive halving: cheap screening first, the gentlest survivors on ever more scenarios.
    rounds = [(6, 10), (24, 4), (args.scenarios, 2)]
    alive = dict(candidates)
    for budget, keep in rounds:
        study.log(f"Phase 2 round: {len(alive)} candidates on {budget} scenarios, keeping {keep}")
        scored = []
        for name, params in alive.items():
            results = study.evaluate(name, params, list(range(budget)))
            if all(r["passed"] for r in results):
                scored.append((stress_score(results), name))
            else:
                study.log(f"{name} out")
        scored.sort()
        alive = {name: alive[name] for _, name in scored[:keep]}
        study.log("Round leaders: " + ", ".join(f"{name} score {score:.0f}" for score, name in scored[:keep]))
        if not alive:
            break
    study.log(f"Current settings on the development scenarios: {summary(baseline)}")
    for name in alive:
        study.log(
            f"Best: {name} {alive[name]}: {summary(list(study.state['runs'][name]['results'].values()))}"
        )
    if args.holdout and alive:
        held_out = list(range(args.holdout[0], args.holdout[1] + 1))
        study.log(f"Held-out check on scenarios {held_out[0]}-{held_out[-1]}")
        for name, params in [("current", DEFAULTS), *alive.items()]:
            study.log(f"Held out, {name}: {summary(study.evaluate(name + '_holdout', params, held_out))}")
    study.log("Done")


if __name__ == "__main__":
    main()
