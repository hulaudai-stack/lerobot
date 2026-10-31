# SO-101 automatic calibration in simulation

Tests `lerobot-calibrate --robot.auto_calibrate=true` against a physics simulation of the SO-101 follower, clamped to a table with a wall behind it, before running it on a real arm.

- **Robot:** the official SO-101 MuJoCo model from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), downloaded on first use. Its joint ranges are the mechanical stops the calibration must find.
- **Servos:** each joint is driven by an emulated Feetech STS3215:
  - a 12-bit encoder mounted at a random angle and direction, as on a real assembly;
  - a homing offset with wrap-around;
  - position limits and a torque limit;
  - a position controller whose stiffness matches the 29-tick sag measured on a real SO-101 elbow.
- **Code under test:** LeRobot's own `SO101Follower` and `SerialMotorsBus` code runs unmodified over a fake serial bus. Each `time.sleep` of that code advances the physics.

## Setup

From the root of your LeRobot checkout, with its environment active:

```bash
uv pip install mujoco matplotlib
```

## Run

```bash
cd examples/so101_auto_calibration_sim

# One calibration, with a video and a found-vs-true report (wall 4 cm behind the base by default)
python run_calibration.py --seed 0 --video out/run.mp4 --log out/run.log

# Many random assemblies and wall distances, two at a time
python batch.py out/batch 1 8 0.04 0.10

# Which stops can be reached at all without touching the table, the wall or the arm itself
python reachability.py
```

On Linux without a display, set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (software).

## Watch it live

MuJoCo's interactive viewer runs the same calibration in a 3D window. Drag to rotate, right-drag to pan, scroll to zoom. On macOS it must be started with `mjpython`, which comes with `mujoco`:

```bash
mjpython viewer.py --seed 0 --speed 2
```

## The wall

The wall must leave the shoulder room to retract fully: with the arm folded, that takes 3.0 cm behind the base, so the default is 4 cm. Closer than that, no posture lets the shoulder reach its back stop straight behind it, and its range comes out short.
