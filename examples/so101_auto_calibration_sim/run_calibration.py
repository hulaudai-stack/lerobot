"""Run LeRobot's SO-101 automatic calibration against the simulated arm and score it.

Usage: python run_calibration.py [--seed N] [--wall-gap M] [--video PATH] [--start-jitter DEG]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101sim import JOINTS, CaptionHandler, SimArm, SimBus, build_model, random_servos, write_video  # noqa: E402

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402

MARGIN = 20  # ticks the calibration keeps inside each stop


def run(
    seed: int, wall_gap: float | None, video: Path | None, start_jitter_deg: float, record_every_s: float
):
    rng = np.random.default_rng(seed)
    model = build_model(wall_gap=wall_gap)
    arm = SimArm(model, random_servos(rng), frame_every_s=record_every_s, record=video is not None)
    # Start roughly in the middle pose, as a person would put it.
    for joint in JOINTS:
        arm.data.qpos[arm.qadr[joint]] = np.radians(rng.uniform(-start_jitter_deg, start_jitter_deg))
    arm.data.qpos[arm.qadr["gripper"]] = np.radians(rng.uniform(5, 30))
    mujoco.mj_forward(model, arm.data)
    if arm.contacts():
        raise RuntimeError(f"start pose touches something: {arm.contacts()}")

    handler = CaptionHandler(arm)
    bus_logger = logging.getLogger("lerobot.motors.motors_bus")
    bus_logger.addHandler(handler)
    bus_logger.setLevel(logging.INFO)

    calibration_dir = Path(tempfile.mkdtemp())
    config = SO101FollowerConfig(
        port="/dev/sim", id=f"sim{seed}", calibration_dir=calibration_dir, auto_calibrate=True
    )
    robot = SO101Follower(config)
    robot.bus._bus = SimBus(arm, robot.bus._definitions)

    error = None
    started = time.time()
    with (
        patch("lerobot.motors.motors_bus.time.sleep", arm.sleep),
        patch("builtins.input", lambda prompt="": ""),
        patch("builtins.print", lambda *a, **k: None),
    ):
        try:
            robot.calibrate()
        except Exception as exc:  # report it with what was found so far
            error = f"{type(exc).__name__}: {exc}"
    bus_logger.removeHandler(handler)

    results = {}
    if error is None:
        for joint in JOINTS:
            if joint == "wrist_roll":
                continue
            cal = robot.calibration[joint]
            jid = model.joint(joint).id
            true_low, true_high = np.degrees(model.jnt_range[jid])
            ends = [
                np.degrees(arm.angle_of_reading(joint, cal.range_min - MARGIN, cal.homing_offset)),
                np.degrees(arm.angle_of_reading(joint, cal.range_max + MARGIN, cal.homing_offset)),
            ]
            found_low, found_high = sorted(ends)
            results[joint] = {
                "true": [round(true_low, 1), round(true_high, 1)],
                "found": [round(found_low, 1), round(found_high, 1)],
                "short_deg": [round(found_low - true_low, 1), round(true_high - found_high, 1)],
            }
    summary = {
        "seed": seed,
        "wall_gap": wall_gap,
        "error": error,
        "sim_seconds": round(arm.time_s, 1),
        "wall_seconds": round(time.time() - started, 1),
        "servos": {j: [round(s.mount), s.direction] for j, s in arm.servos.items()},
        "results": results,
        "log_tail": handler.lines[-12:],
    }
    if video is not None:
        np.save(video.with_suffix(".trace.npy"), np.array(arm.trace))
    if video is not None and arm.frames:
        write_video(arm.frames, video)
    return summary, handler.lines, arm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wall-gap", type=float, default=0.04)
    parser.add_argument("--no-wall", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start-jitter", type=float, default=8.0)
    parser.add_argument("--record-every", type=float, default=0.25)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    summary, lines, _ = run(
        args.seed, None if args.no_wall else args.wall_gap, args.video, args.start_jitter, args.record_every
    )
    if args.log:
        args.log.write_text("\n".join(lines))
    print(json.dumps(summary, indent=1))
