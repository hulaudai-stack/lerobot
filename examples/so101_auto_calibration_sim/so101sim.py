"""SO-101 physics simulation for testing LeRobot's automatic calibration.

The official SO-101 MuJoCo model (TheRobotStudio/SO-ARM100) is placed on a table, held by two
C-clamps, with a wall behind it. Each joint is driven by an emulated Feetech STS3215: a 12-bit
encoder mounted at a random angle and direction, a homing offset with wrap-around, position
limits, a torque limit and a position controller. LeRobot's own `SOFollower` and
`SerialMotorsBus` code drives it through a fake serial bus, and each `time.sleep` of that code
advances the physics.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
# The LeRobot checkout whose code is tested: this folder's repository, or LEROBOT_ROOT.
LEROBOT_ROOT = Path(
    os.environ.get("LEROBOT_ROOT", next((p for p in HERE.parents if (p / "src/lerobot").is_dir()), HERE))
)
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))  # its lerobot, not whichever is installed
from tests.mocks.mock_motors_bus import MockBus  # noqa: E402

MODEL_REPO = "https://github.com/TheRobotStudio/SO-ARM100.git"
MODEL_DIR = Path(os.environ.get("SO_ARM100_DIR", HERE / "SO-ARM100"))
ROBOT_XML = MODEL_DIR / "Simulation/SO101/so101_new_calib.xml"


def fetch_model() -> None:
    """Download the official SO-101 model (TheRobotStudio/SO-ARM100) on first use."""
    if not ROBOT_XML.exists():
        subprocess.run(["git", "clone", "--depth", "1", MODEL_REPO, str(MODEL_DIR)], check=True)


TICKS = 4096
TICKS_PER_RAD = TICKS / (2 * math.pi)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IDS = {name: index + 1 for index, name in enumerate(JOINTS)}
MAX_TORQUE_NM = 3.35  # stall torque of the actuators in the model
STRESS_PCT = 20  # % of the servo's maximum torque above which a joint counts as pushing hard
SERVO_KP = 10.0  # N.m/rad
SERVO_KV = 0.3  # N.m.s/rad
BASE_BACK_X = -0.031  # back edge of the base, in the model's frame
TABLE_TOP_Z = -0.0024  # underside of the base


def build_model(wall_gap: float | None = 0.04, clamps: bool = True) -> mujoco.MjModel:
    """The SO-101 on a table, clamped, with a wall `wall_gap` metres behind its base (no wall
    for None). The base gets collision hulls, which the official model leaves out."""
    fetch_model()
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.option.timestep = 0.002
    base = spec.body("base")
    for geom in list(base.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname != "waveshare_mounting_plate_so101_v2":
            hull = base.add_geom()
            hull.type, hull.meshname = mujoco.mjtGeom.mjGEOM_MESH, geom.meshname
            hull.pos, hull.quat = geom.pos, geom.quat
            hull.contype, hull.conaffinity, hull.group = 1, 1, 3
            hull.rgba = [0, 0, 0, 0]

    world = spec.worldbody
    wood, white, steel = [0.62, 0.45, 0.28, 1], [0.92, 0.92, 0.9, 1], [0.45, 0.47, 0.5, 1]
    back = BASE_BACK_X - (wall_gap if wall_gap is not None else 0.30)
    front = 0.70
    table = world.add_geom()
    table.name, table.type = "table", mujoco.mjtGeom.mjGEOM_BOX
    table.size = [(front - back) / 2, 0.6, 0.02]
    table.pos = [(front + back) / 2, 0, TABLE_TOP_Z - 0.02]
    table.rgba = wood
    if wall_gap is not None:
        wall = world.add_geom()
        wall.name, wall.type = "wall", mujoco.mjtGeom.mjGEOM_BOX
        wall.size = [0.01, 0.6, 0.45]
        wall.pos = [back - 0.01, 0, TABLE_TOP_Z + 0.45]
        wall.rgba = white
    if clamps:
        for side in (-1, 1):
            # The C-clamp's upper jaw and screw pad, pressing the base's side flange to the table.
            jaw = world.add_geom()
            jaw.name, jaw.type = f"clamp_{'left' if side > 0 else 'right'}", mujoco.mjtGeom.mjGEOM_BOX
            jaw.size = [0.03, 0.012, 0.012]
            jaw.pos = [0.02, side * 0.068, TABLE_TOP_Z + 0.012]
            jaw.rgba = steel
    # The pan bearing joins the base and the shoulder; welded to the world, the base escapes
    # MuJoCo's parent-child contact filter, so the pair is excluded explicitly.
    exclude = spec.add_exclude()
    exclude.bodyname1, exclude.bodyname2 = "base", "shoulder"
    sky = spec.add_texture()
    sky.name, sky.type, sky.builtin = (
        "sky",
        mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    )
    sky.rgb1, sky.rgb2, sky.width, sky.height = [0.55, 0.65, 0.78], [0.12, 0.14, 0.18], 256, 1536
    light = world.add_light()
    light.pos, light.dir = [0.3, -0.6, 1.2], [-0.2, 0.5, -1]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    model = spec.compile()
    # STS3215 with LeRobot's P gain of 16: an effective stiffness matching the 29-tick sag measured
    # on a real SO-101 elbow holding the forearm level (0.45 N.m of gravity torque).
    model.actuator_gainprm[:, 0] = SERVO_KP
    model.actuator_biasprm[:, 1] = -SERVO_KP
    model.actuator_biasprm[:, 2] = -SERVO_KV
    # A servo told to go past a stop pushes into it, up to its torque limit: its target is not
    # clipped to the joint's range, as the official model's ctrlrange would have it.
    model.actuator_ctrllimited[:] = 0
    return model


@dataclass
class Servo:
    """How a servo is mounted: the raw encoder reading at joint angle 0, and whether the reading
    grows with the angle (+1) or against it (-1)."""

    mount: float
    direction: int


@dataclass
class SimArm:
    model: mujoco.MjModel
    servos: dict[str, Servo]
    frame_every_s: float = 0.1
    width: int = 480
    height: int = 360
    record: bool = True
    noise_ticks: int = 0  # each reading off by up to this many ticks, as a real encoder jitters
    data: mujoco.MjData = field(init=False)
    bus: MockBus | None = field(default=None, init=False)
    time_s: float = field(default=0.0, init=False)
    frames: list[np.ndarray] = field(default_factory=list, init=False)
    trace: list[tuple] = field(default_factory=list, init=False)
    trace_every_s: float = 0.02
    caption: str = field(default="", init=False)
    _next_frame_s: float = field(default=0.0, init=False)
    _renderer: mujoco.Renderer | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.data = mujoco.MjData(self.model)
        # Stress on each joint: the largest torque its servo applied, and how long it applied more
        # than STRESS_PCT of its maximum, the time spent pushing hard rather than holding the arm.
        self._peak = np.zeros(len(JOINTS))
        self._hard = np.zeros(len(JOINTS))
        self.joint_ids = {name: self.model.joint(name).id for name in JOINTS}
        self.actuator_ids = {name: self.model.actuator(name).id for name in JOINTS}
        self.qadr = {name: self.model.jnt_qposadr[self.joint_ids[name]] for name in JOINTS}
        self._actuators = np.array([self.actuator_ids[name] for name in JOINTS])

    @property
    def peak_torque(self) -> dict[str, float]:
        """The largest torque each joint's servo applied, in N.m."""
        return dict(zip(JOINTS, self._peak.tolist(), strict=True))

    @property
    def hard_push_s(self) -> dict[str, float]:
        """How long each joint's servo applied more than STRESS_PCT of its maximum torque."""
        return dict(zip(JOINTS, self._hard.tolist(), strict=True))

    # -- encoder ---------------------------------------------------------------------------

    def angle(self, joint: str) -> float:
        return float(self.data.qpos[self.qadr[joint]])

    def raw(self, joint: str) -> float:
        servo = self.servos[joint]
        return servo.mount + servo.direction * self.angle(joint) * TICKS_PER_RAD

    def load(self, joint: str) -> int:
        """Present_Load as an STS3215 reports it: the torque applied, in 0.1% of its maximum,
        signed towards higher readings."""
        torque = float(self.data.actuator_force[self.actuator_ids[joint]])
        return int(round(self.servos[joint].direction * torque / MAX_TORQUE_NM * 1000))

    def _register(self, joint: str, name: str) -> int:
        assert self.bus is not None
        return self.bus.stored(IDS[joint], name)

    def present(self, joint: str) -> float:
        """The reading: Feetech subtracts the homing offset, modulo the encoder."""
        return (self.raw(joint) - self._register(joint, "homing_offset")) % TICKS

    def angle_of_reading(self, joint: str, reading: float, homing_offset: int) -> float:
        """The joint angle at which the encoder reads `reading` with `homing_offset`, taking the
        turn nearest the joint's middle."""
        servo = self.servos[joint]
        raw = reading + homing_offset - servo.mount
        raw = (raw + TICKS / 2) % TICKS - TICKS / 2
        return servo.direction * raw / TICKS_PER_RAD

    # -- servo control ---------------------------------------------------------------------

    def _drive(self) -> None:
        for joint in JOINTS:
            actuator = self.actuator_ids[joint]
            on = self._register(joint, "torque_enable")
            limit = self._register(joint, "torque_limit") if on else 0
            force = MAX_TORQUE_NM * limit / 1000
            self.model.actuator_forcerange[actuator] = [-force, force]
            low, high = (
                self._register(joint, "min_position_limit"),
                self._register(joint, "max_position_limit"),
            )
            goal = min(high, max(low, self._register(joint, "goal_position")))
            # The servo turns by the difference between its goal and its reading, never across the
            # wrap of its encoder: it moves within its 0..4095 readings.
            error = goal - self.present(joint)
            target = self.angle(joint) + self.servos[joint].direction * error / TICKS_PER_RAD
            self.data.ctrl[actuator] = target if on else self.angle(joint)

    def sleep(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            self._drive()
            mujoco.mj_step(self.model, self.data)
            self.time_s += self.model.opt.timestep
            torque = np.abs(self.data.actuator_force[self._actuators])
            np.maximum(self._peak, torque, out=self._peak)
            self._hard += (torque > STRESS_PCT / 100 * MAX_TORQUE_NM) * self.model.opt.timestep
            if not self.trace or self.time_s - self.trace[-1][0] >= self.trace_every_s:
                self.trace.append(
                    (
                        self.time_s,
                        *(self.angle(j) for j in JOINTS),
                        *(self.present(j) for j in JOINTS),
                        *(self._register(j, "goal_position") for j in JOINTS),
                        *(
                            self._register(j, "torque_limit") * self._register(j, "torque_enable")
                            for j in JOINTS
                        ),
                    )
                )
            if self.record and self.time_s >= self._next_frame_s:
                self.frames.append(self.render())
                self._next_frame_s += self.frame_every_s

    # -- pictures --------------------------------------------------------------------------

    def render(self, azimuth: float = 125, elevation: float = -18, distance: float = 1.0) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
            self._options = mujoco.MjvOption()
            self._options.geomgroup[3] = 0
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.08, 0.0, 0.16]
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        self._renderer.update_scene(self.data, camera, self._options)
        image = self._renderer.render().copy()
        return _caption(image, f"t={self.time_s:6.1f}s  {self.caption}")

    def contacts(self) -> list[tuple[str, str]]:
        names = []
        for contact in self.data.contact[: self.data.ncon]:
            a, b = (
                self.model.body(self.model.geom_bodyid[g]).name or self.model.geom(g).name
                for g in (contact.geom1, contact.geom2)
            )
            a = self.model.geom(contact.geom1).name or a
            b = self.model.geom(contact.geom2).name or b
            names.append((a, b))
        return names


def _caption(image: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return image
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    draw.rectangle([0, 0, picture.width, 22], fill=(20, 20, 20))
    draw.text((6, 5), text, fill=(240, 240, 240))
    return np.asarray(picture)


class SimBus(MockBus):
    """A `MockBus` whose present positions come from the simulated encoders."""

    def __init__(self, arm: SimArm, motors):
        super().__init__(motors=motors)
        self.arm = arm
        self.rng = np.random.default_rng(0)
        arm.bus = self
        for joint, id_ in IDS.items():
            self.seed(id_, "model_number", 777)
            self.seed(id_, "torque_limit", 1000)
            self.seed(id_, "min_position_limit", 0)
            self.seed(id_, "max_position_limit", TICKS - 1)
            self.seed(id_, "goal_position", round(arm.present(joint)))

    def _refresh(self) -> None:
        jitter = self.arm.noise_ticks
        for joint, id_ in IDS.items():
            noise = int(self.rng.integers(-jitter, jitter + 1)) if jitter else 0
            self.seed(id_, "present_position", (int(round(self.arm.present(joint))) + noise) % TICKS)
            self.seed(id_, "present_load", self.arm.load(joint))

    def read_register_with_error(self, motor_id, register, retries=0):
        self._refresh()
        return super().read_register_with_error(motor_id, register, retries)

    def sync_read_register_with_error(self, motor_ids, register, retries=0):
        self._refresh()
        return super().sync_read_register_with_error(motor_ids, register, retries)


def random_servos(rng: np.random.Generator) -> dict[str, Servo]:
    """Servos mounted as an assembly does: centred near 2048 when the joint is in the middle,
    give or take a horn spline, either way round."""
    return {joint: Servo(2048 + rng.uniform(-300, 300), int(rng.choice([-1, 1]))) for joint in JOINTS}


def write_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    process.wait()


class CaptionHandler(logging.Handler):
    """Shows the calibration's log lines in the video captions."""

    def __init__(self, arm: SimArm):
        super().__init__()
        self.arm = arm
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        self.lines.append(f"[{self.arm.time_s:7.1f}s] {message}")
        self.arm.caption = message[:90]
