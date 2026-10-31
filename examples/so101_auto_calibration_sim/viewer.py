"""Watch LeRobot's SO-101 automatic calibration live in MuJoCo's interactive 3D viewer.

On macOS the viewer must be started with `mjpython` (installed with the `mujoco` package):

    mjpython viewer.py [--seed N] [--wall-gap 0.04] [--speed 1.0]

Drag to rotate, right-drag to pan, scroll to zoom. The calibration's log lines print in the terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from so101sim import JOINTS, SimArm, SimBus, build_model, random_servos  # noqa: E402

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wall-gap", type=float, default=0.04)
    parser.add_argument("--speed", type=float, default=1.0, help="1 = real time, 4 = four times faster")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = build_model(wall_gap=args.wall_gap)
    arm = SimArm(model, random_servos(rng), record=False)
    for joint in JOINTS:
        arm.data.qpos[arm.qadr[joint]] = np.radians(rng.uniform(-10, 10))
    arm.data.qpos[arm.qadr["gripper"]] = np.radians(15)
    mujoco.mj_forward(model, arm.data)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = SO101FollowerConfig(
        port="/dev/sim",
        id=f"viewer{args.seed}",
        calibration_dir=Path(tempfile.mkdtemp()),
        auto_calibrate=True,
    )
    robot = SO101Follower(config)
    robot.bus._bus = SimBus(arm, robot.bus._definitions)

    with mujoco.viewer.launch_passive(model, arm.data) as viewer:
        viewer.cam.lookat[:] = [0.08, 0.0, 0.16]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.0, 125, -18
        step = arm.sleep

        def paced_sleep(seconds: float) -> None:
            started = time.perf_counter()
            step(seconds)
            viewer.sync()
            if not viewer.is_running():
                raise KeyboardInterrupt
            time.sleep(max(0.0, seconds / args.speed - (time.perf_counter() - started)))

        with (
            patch("lerobot.motors.motors_bus.time.sleep", paced_sleep),
            patch("builtins.input", lambda p="": ""),
        ):
            robot.calibrate()
        print("Calibration finished:")
        for joint, cal in robot.calibration.items():
            print(
                f"  {joint:14s} range {cal.range_min:5d}..{cal.range_max:5d}  homing offset {cal.homing_offset}"
            )
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
