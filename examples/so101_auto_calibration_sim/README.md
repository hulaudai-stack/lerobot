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

On Linux without a display, the scripts use software rendering (`MUJOCO_GL=osmesa`); set `MUJOCO_GL=egl` for a GPU. On macOS nothing needs setting.

## Watch it live

MuJoCo's interactive viewer runs the same calibration in a 3D window. Drag to rotate, right-drag to pan, scroll to zoom. On macOS it must be started with `mjpython`, which comes with `mujoco`:

```bash
mjpython viewer.py --seed 0 --speed 2
```

## The wall

The wall must leave the shoulder room to retract fully: with the arm folded, that takes 3.0 cm behind the base, so the default is 4 cm. Closer than that, no posture lets the shoulder reach its back stop straight behind it, and its range comes out short.

## What the simulation found

Every result below is from LeRobot's own calibration code driving the simulated arm.

| Change | Before | After |
|---|---|---|
| Let the goal lead by 200 ticks, so a push reaches the torque limit | Shoulder and elbow too weak to lift: 2 of 5 joints right | All 5 joints right |
| Re-centre the range after every posture; keep goals 150 ticks from the encoder's ends | A joint falling with gravity read across the end of its encoder | No wraps |
| Pan with the upper arm upright, forearm raised, wrist straight | Hand swept into the wall at ±105° | Full ±110° |
| Followers track the joint's position through a ratchet, not its goal | Elbow wedged the hand into the table behind (4 of 8 at 20 cm) | 5 of 5 |
| Revisit the elbow with the upper arm upright and the wrist bent each way | Elbow fold 4–5° short when the shoulder started tilted back | Within 1° |
| Re-survey a joint whose start-pose survey was unclear | Wrong "up" for the elbow: hand dragged along the table | Correct |
| Read gravity from the sag of a held joint before turning it | A hand resting on the table hid the pull | Correct |
| Tip the shoulder back with the wrist straight, not bent up | 3.5 cm wall: hand curled into it, shoulder 23° short | Full range |
| Let go of a joint shoved by another's push before that one turns back | Shoved shoulder sprang back and wedged the hand into the wall at 100% torque | No wedge |
| Cap moves at 45% torque, full torque only if blocked | Return moves pressed the hand into the wall at up to 33 N | ≤ 23 N, 5.5 N mean |
| Raise only the forearm at the start; flip a raise blocked at once | Raising the shoulder "upright" leant the arm back into the wall | Correct |
| Shoulder sweeps with elbow and wrist blending to targets found by `posture_search.py` | 2 postures for the shoulder | 1 in most runs |
| Stop searching a joint once its range is as wide as designed (`expected_ranges`) | Every posture and revisit always run | 141 s → 123 s |

From a start roughly in the middle pose (within 20° per joint, 30° for the pan), with a random servo assembly and a wall 3.5 cm to 20 cm behind the base or none:

| Scenarios | Encoder noise | Every stop within 3° |
|---|---|---|
| 0–47, used while developing | none | 48 / 48 |
| 48–95, fresh | ±1 tick | 48 / 48 |
| 96–143, held out until the end | ±1 tick | 47 / 48 |

That is 143 of 144. With the changes below the table, every scenario passes:

| Scenarios | Pass | Peak torque, mean / max | Env. contact, mean / max | Steps | Arm time |
|---|---|---|---|---|---|
| 744–1343, before | 599 / 600 | 60% / 100% | 7.2 N / 21 N | 27 pushes + 69 moves | 174 s |
| 90 hardest (all past failures, highest forces, 0–47) | 90 / 90 | 40% / 47% | 6.1 N / 19.5 N | 23 pushes + 65 moves | 124 s |
| 1344–1943, fresh | **600 / 600** | 40% / 47% | 5.4 N / 22.8 N | 22 pushes + 65 moves | 122 s |
| 744–1343, now | **600 / 600** | 40% / 46% | 5.7 N / 23.7 N | 22 pushes + 65 moves | 123 s |

The settings LeRobot ships were chosen by comparing variants on the 90 hardest scenarios with `optimize.py`: a torque ladder of 15% then 30%, two shoulder postures instead of four, moves capped at 45%. A single push at 30% (`--robot.auto_calibration_torque_limits=[300]`) saves another 6 pushes and 7 s at the same forces (599 of 600, before the fix of the one failure, a hand resting on the table), but presses every stop at 30% and has no stronger push to tell a stop from a joint too weak to move on.

### Limits

- **Start far from the middle pose.** Started anywhere within ~45% of each joint's range, 35 of 48 runs pass (30 before). Most failures are a hand that starts close to the table getting wedged against it. Recovering would mean dropping torque on a loaded joint, which on a real arm drops the shoulder, so the calibration asks to start in the middle pose.
- **Contact.** Every run touches the table or the wall: a stop the arm cannot reach without touching either is reached by pressing on it, at a reduced torque. About one run in six presses on it with more than 10 N, briefly.
- **Speed.** Ramping faster than 8 ticks per 20 ms fails even for moves back through space already swept: a hand starting near the table swings into it and wedges. The same holds for a first push above 15% torque. The optimiser therefore only tunes stall detection, survey size, the torque ladder and the postures.
