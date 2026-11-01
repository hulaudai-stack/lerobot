#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from unittest.mock import call, patch

import pytest

from lerobot.motors.motors_bus import (
    Follow,
    Motor,
    MotorCalibration,
    MotorNormMode,
    Raised,
    SerialMotorsBus,
    Toward,
)
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _require_package_cache
from tests.mocks.mock_motors_bus import DUMMY_1, DUMMY_2, DUMMY_3, MockBus, MockMotorsBus


@pytest.fixture
def dummy_motors() -> dict[str, Motor]:
    return {
        "dummy_1": Motor(1, "model_2", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "model_3", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "model_2", MotorNormMode.RANGE_0_100),
    }


@pytest.fixture
def mixed_motors() -> dict[str, Motor]:
    """Two servo definitions on one bus."""
    return {
        "dummy_1": Motor(1, "model_1", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "model_2", MotorNormMode.RANGE_M100_100),
    }


def test_register_names_resolve_without_case():
    bus = MockMotorsBus("/dev/dummy-port", {"dummy": Motor(1, "model_1", MotorNormMode.RANGE_M100_100)})

    assert bus._has_register("dummy", "Firmware_Version")
    assert bus._has_register("dummy", "firmware_version")
    assert not bus._has_register("dummy", "Lock")


def test_unknown_model_is_refused():
    pytest.importorskip("rustypot")
    with pytest.raises(ValueError, match="Unknown motor model 'sts9999'"):
        SerialMotorsBus("/dev/dummy-port", {"dummy": Motor(1, "sts9999", MotorNormMode.RANGE_M100_100)})


def test_a_missing_rustypot_points_to_the_serial_motors_extra():
    with patch("lerobot.utils.import_utils.is_package_available", return_value=False):
        _require_package_cache.clear()
        try:
            with pytest.raises(ImportError, match=r"lerobot\[serial-motors\]"):
                SerialMotorsBus(
                    "/dev/dummy-port", {"dummy": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)}
                )
        finally:
            _require_package_cache.clear()


def test_connect_opens_a_bus_with_each_motor_definition(mixed_motors):
    bus = MockMotorsBus("/dev/dummy-port", mixed_motors)
    bus.connect(handshake=False)

    # rustypot counts the timeout in seconds, LeRobot in milliseconds.
    assert (bus._bus.serial_port, bus._bus.baudrate, bus._bus.timeout) == (
        "/dev/dummy-port",
        bus.default_baudrate,
        bus.default_timeout / 1000,
    )
    assert bus._bus.motors == {1: DUMMY_1, 2: DUMMY_2}


@pytest.fixture
def bus(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    return bus


@pytest.mark.parametrize(
    "data_name, id_, value, normalized",
    [
        ("Firmware_Version", 1, 14, False),
        ("Model_Number", 1, 5678, False),
        ("Present_Position", 2, 1337, True),
        ("present_position", 2, 1337, True),
        ("Present_Velocity", 3, -1337, False),
    ],
)
def test_read(data_name, id_, value, normalized, bus):
    bus._bus.seed(id_, data_name.lower(), value)

    with patch.object(MockMotorsBus, "_normalize", return_value={id_: value}) as mock__normalize:
        assert bus.read(data_name, f"dummy_{id_}") == value

    assert bus._bus.reads == [(id_, data_name.lower())]
    assert mock__normalize.call_args_list == ([call({id_: value})] if normalized else [])


ACCESSES = pytest.mark.parametrize(
    "access",
    [
        lambda bus: bus.read("Present_Position", "dummy_1", normalize=False),
        lambda bus: bus.write("Goal_Position", "dummy_1", 1337, normalize=False),
        lambda bus: bus.sync_read("Present_Position", normalize=False),
    ],
    ids=["read", "write", "sync_read"],
)


@ACCESSES
def test_a_motor_refusing_the_access_fails_it(access, bus):
    """On protocol 2, an error number in the status is a refusal: 7, an access error."""
    bus._bus.status[1] = 0x87

    with pytest.raises(RuntimeError, match="motor ID 1 refused it with error status 0x87"):
        access(bus)


@ACCESSES
def test_an_alarm_in_the_status_does_not_fail_the_access(access, bus):
    """The alert bit alone reports a latched hardware fault, not a refused access."""
    bus._bus.status[1] = 0x80

    access(bus)


@pytest.mark.parametrize(
    "data_name, id_, value",
    [
        ("Goal_Position", 1, 1337),
        ("Goal_Velocity", 2, -1337),
        ("Lock", 3, 1),
    ],
)
def test_write(data_name, id_, value, bus):
    with patch.object(MockMotorsBus, "_unnormalize", return_value={id_: value}) as mock__unnormalize:
        bus.write(data_name, f"dummy_{id_}", value)

    assert bus._bus.writes == [(id_, data_name.lower(), value)]
    if data_name.lower() in bus.normalized_data:
        mock__unnormalize.assert_called_once_with({id_: value})


@pytest.mark.parametrize(
    "motors, ids_values",
    [
        ("dummy_1", {1: 5678}),
        (["dummy_1", "dummy_2"], {1: 1337, 2: 42}),
        (None, {1: 1337, 2: -42, 3: 4016}),
    ],
    ids=["by name", "by list", "every motor"],
)
@pytest.mark.parametrize("data_name", ["Model_Number", "Present_Position", "Present_Velocity"])
def test_sync_read(data_name, motors, ids_values, bus):
    for id_, value in ids_values.items():
        bus._bus.seed(id_, data_name.lower(), value)

    with patch.object(MockMotorsBus, "_normalize", return_value=ids_values) as mock__normalize:
        returned = bus.sync_read(data_name, motors)

    assert returned == {f"dummy_{id_}": value for id_, value in ids_values.items()}
    assert bus._bus.sync_reads == [(list(ids_values), data_name.lower())]
    if data_name.lower() in bus.normalized_data:
        mock__normalize.assert_called_once_with(ids_values)


@pytest.mark.parametrize(
    "values, ids_values",
    [
        (500, {1: 500, 2: 500, 3: 500}),
        ({"dummy_1": 1337, "dummy_2": -42, "dummy_3": 4016}, {1: 1337, 2: -42, 3: 4016}),
    ],
    ids=["one value for every motor", "a value per motor"],
)
@pytest.mark.parametrize("data_name", ["Goal_Position", "Goal_Velocity", "Lock"])
def test_sync_write(data_name, values, ids_values, bus):
    with patch.object(MockMotorsBus, "_unnormalize", return_value=ids_values) as mock__unnormalize:
        bus.sync_write(data_name, values)

    assert bus._bus.sync_writes == [(list(ids_values), data_name.lower(), list(ids_values.values()))]
    if data_name.lower() in bus.normalized_data:
        mock__unnormalize.assert_called_once()


@pytest.mark.parametrize(
    "access",
    [
        lambda bus: bus.read("Present_Position", "dummy_1", normalize=False),
        lambda bus: bus.write("Goal_Position", "dummy_1", 1337, normalize=False),
        lambda bus: bus.sync_read("Present_Position", normalize=False),
    ],
    ids=["read", "write", "sync_read"],
)
def test_a_motor_that_does_not_answer_is_a_connection_error(access, bus):
    bus._bus.absent.add(1)

    with pytest.raises(ConnectionError, match="Timeout"):
        access(bus)


def test_a_sync_write_fails_only_when_the_port_does(bus):
    """No motor answers a sync write, so one that is absent goes unnoticed."""
    bus._bus.absent.add(1)
    bus.sync_write("Goal_Position", 1337, normalize=False)

    with (
        patch.object(MockBus, "sync_write_register", side_effect=OSError("port closed")),
        pytest.raises(ConnectionError, match="port closed"),
    ):
        bus.sync_write("Goal_Position", 1337, normalize=False)


def test_retries_are_left_to_rustypot(bus):
    bus.read("Present_Position", "dummy_1", normalize=False, num_retry=1)
    bus.write("Goal_Position", "dummy_1", 1337, normalize=False, num_retry=2)
    bus.sync_read("Present_Position", "dummy_1", normalize=False, num_retry=3)
    bus.sync_write("Goal_Position", {"dummy_1": 1337}, normalize=False, num_retry=4)

    assert bus._bus.retries == [1, 2, 3, 4]


def test_ping(bus):
    bus._bus.seed(2, "model_number", 5678)

    assert bus.ping(2) == 5678

    bus._bus.absent.add(3)
    assert bus.ping(3) is None

    bus._bus.status[2] = 0x20
    assert bus.ping(2) is None


def test_a_motor_is_found_with_a_broadcast_ping_where_its_servo_answers_one(bus):
    bus._bus.seed(9, "model_number", 5678)

    assert bus._find_single_motor("dummy_3") == (1_000_000, 9)
    assert (bus._bus.broadcast_scans, bus._bus.scans) == ([DUMMY_2], [])


def test_a_motor_whose_servo_answers_no_broadcast_ping_is_found_with_a_sweep():
    bus = MockMotorsBus("/dev/dummy-port", {"dummy": Motor(1, "model_1", MotorNormMode.RANGE_M100_100)})
    bus.connect(handshake=False)
    bus._bus.seed(9, "model_number", 1234)

    assert bus._find_single_motor("dummy") == (1_000_000, 9)
    assert (bus._bus.broadcast_scans, bus._bus.scans) == ([], [DUMMY_1])


def test_setup_refuses_a_motor_of_another_model(bus):
    bus._bus.seed(9, "model_number", 5799)

    with pytest.raises(RuntimeError, match="different than the one expected: 5678"):
        bus._find_single_motor("dummy_3")


def test_the_motor_search_tries_the_default_baudrate_then_the_factory_one(bus):
    """A motor already set up answers on the first try, a new one on the second; then the
    rest, in a fixed order."""
    with (
        patch.object(MockBus, "set_baudrate", autospec=True) as set_baudrate,
        pytest.raises(RuntimeError, match="was not found"),
    ):
        bus._find_single_motor("dummy_3")

    assert [call.args[1] for call in set_baudrate.call_args_list] == [1_000_000, 500_000, 250_000]


def test_setup_motor_gives_the_motor_its_id_and_the_default_baudrate(dummy_motors):
    """A new motor answers at an id the bus does not have (9 here); rustypot reaches it
    there through its definition."""
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    bus._bus.seed(9, "model_number", 5678)

    bus.setup_motor("dummy_3", initial_baudrate=500_000, initial_id=9)

    assert bus._bus.setups == [
        ("change_id", DUMMY_2, 9, 3),
        ("change_baudrate", DUMMY_2, 3, 1_000_000),
    ]
    assert len(bus.opened) == 1
    assert bus._bus.baudrate == bus.default_baudrate


def test_a_failed_setup_puts_the_port_back_at_the_default_baudrate(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    bus._bus.absent.add(9)

    with pytest.raises(ConnectionError, match="'dummy_3'"):
        bus.setup_motor("dummy_3", initial_baudrate=500_000, initial_id=9)

    assert bus.is_connected
    assert bus._bus.baudrate == bus.default_baudrate


def test_a_failed_motor_search_puts_the_port_back_at_the_default_baudrate(bus):
    """The search walks the baud rates; finding nothing must not leave the port on the last."""
    with pytest.raises(RuntimeError, match="was not found"):
        bus.setup_motor("dummy_3")

    assert bus._bus.baudrate == bus.default_baudrate


def test_a_motor_that_does_not_answer_as_set_up_fails_its_setup(bus):
    """The id and baud rate writes are answered before they take effect: the setup reads
    the model number back at the new id and the default baud rate."""
    bus._bus.seed(9, "model_number", 5678)

    # The id write is acknowledged, but the motor keeps its old id.
    with (
        patch.object(MockBus, "change_id", autospec=True),
        pytest.raises(RuntimeError, match=r"'dummy_3' does not answer as motor ID 3"),
    ):
        bus.setup_motor("dummy_3", initial_baudrate=500_000, initial_id=9)


def test_constructing_does_not_open_the_port(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)

    assert not bus.is_connected


def test_connect_reports_a_missing_port(dummy_motors):
    bus = MockMotorsBus("/dev/nope", dummy_motors)

    with (
        patch.object(MockMotorsBus, "_open_bus", side_effect=OSError("no such device")),
        pytest.raises(ConnectionError, match="lerobot-find-port"),
    ):
        bus.connect(handshake=False)

    assert not bus.is_connected


def test_a_failed_handshake_closes_the_port(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)

    with (
        patch.object(MockMotorsBus, "_handshake", side_effect=RuntimeError("motor check failed")),
        pytest.raises(RuntimeError, match="motor check failed"),
    ):
        bus.connect()

    assert not bus.is_connected
    assert bus.opened[-1].closed


def test_a_motor_of_another_model_fails_the_handshake(bus):
    """model_2 and model_3 share a control table, yet the motor set as model_3 must be one."""
    for id_ in bus.ids:
        bus._bus.seed(id_, "model_number", 5678)

    with pytest.raises(RuntimeError, match=r"2 \(dummy_2\): expected 5799, found 5678"):
        bus._handshake()


def test_connect_uses_50ms_timeout(bus):
    assert bus._bus.timeout == 0.05


def test_torque_off_asks_for_every_motor_and_names_the_ones_that_failed(bus):
    """Every motor is attempted; the failures are reported, and a motor that does not answer may
    stay under torque."""
    bus._bus.absent.update({1, 3})

    with pytest.raises(ConnectionError, match=r"'dummy_1' \(motor ID 1\).*'dummy_3' \(motor ID 3\)"):
        bus.disable_torque(num_retry=2)

    assert bus._bus.torques == [([1, 2, 3], False, 2)]
    assert bus._bus.stored(2, "torque_enable") == 0


def test_torque_on_goes_to_the_motors_asked(bus):
    bus.enable_torque(["dummy_1", "dummy_3"])

    assert bus._bus.torques == [([1, 3], True, 0)]


@pytest.mark.parametrize(
    "access",
    [lambda bus: bus.disable_torque(), lambda bus: bus.set_baudrate(57_600), lambda bus: bus.ping(1)],
    ids=["disable_torque", "set_baudrate", "ping"],
)
def test_the_motors_need_a_connected_bus(access, dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)

    with pytest.raises(DeviceNotConnectedError):
        access(bus)


def test_an_operating_mode_is_written_as_the_value_of_its_name(bus):
    bus.set_operating_mode("velocity", ["dummy_1", "dummy_2"])

    assert bus._bus.writes == [(1, "operating_mode", 1), (2, "operating_mode", 1)]


def test_a_mode_a_servo_does_not_have_is_refused_before_any_write(mixed_motors):
    """dummy_2 has the mode and comes first, but dummy_1 has none: no motor changes mode."""
    bus = MockMotorsBus("/dev/dummy-port", mixed_motors)
    bus.connect(handshake=False)

    with pytest.raises(ValueError, match=r"'dummy_1' \(model_1\) has no 'position' operating mode"):
        bus.set_operating_mode("position", ["dummy_2", "dummy_1"])

    assert bus._bus.writes == []


def test_disconnect_closes_the_port_when_torque_off_fails(bus):
    rustypot_bus = bus._bus
    rustypot_bus.absent.add(1)

    with pytest.raises(ConnectionError):
        bus.disconnect()

    assert rustypot_bus.closed
    assert not bus.is_connected


def test_disconnect_closes_the_port(bus):
    rustypot_bus = bus._bus

    bus.disconnect(disable_torque=False)
    bus.disconnect(disable_torque=False)

    assert rustypot_bus.closed
    assert not bus.is_connected


def test_changing_the_baudrate_keeps_the_port_open(bus):
    bus.set_baudrate(57_600)

    assert bus._bus.baudrate == 57_600
    assert bus.is_connected


def test_the_port_reopens_at_the_default_baudrate(bus):
    bus.set_baudrate(57_600)
    bus.disconnect(disable_torque=False)
    bus.connect(handshake=False)

    assert bus._bus.baudrate == bus.default_baudrate


@pytest.fixture
def arm() -> MockMotorsBus:
    """Three calibration-capable motors."""
    bus = MockMotorsBus(
        "/dev/dummy-port",
        {
            "dummy_1": Motor(1, "model_4", MotorNormMode.RANGE_M100_100),
            "dummy_2": Motor(2, "model_4", MotorNormMode.RANGE_M100_100),
            "dummy_3": Motor(3, "model_4", MotorNormMode.RANGE_0_100),
        },
    )
    bus.connect(handshake=False)
    return bus


def test_record_calibration(arm, monkeypatch):
    """The motors are freed, in their mode and direction before the pose is asked; then homed at
    the pose held, ranges swept by hand, and a full-turn motor spans its whole encoder."""
    monkeypatch.setattr(DUMMY_3, "resolution", 1024)
    for id_ in arm.ids:
        arm._bus.seed(id_, "torque_enable", 1)
        arm._bus.seed(id_, "drive_mode", 1)
    poses = [{1: 600, 2: 300, 3: 400}, {1: 100, 3: 200}, {1: 900, 3: 700}]  # held, then swept
    at_prompt = {}

    def move() -> bool:
        if not poses:
            return True
        for id_, position in poses.pop(0).items():
            arm._bus.seed(id_, "present_position", position)
        return False

    def hold_the_pose(prompt: str) -> str:
        registers = ("torque_enable", "operating_mode", "drive_mode")
        at_prompt.update({id_: tuple(arm._bus.stored(id_, r) for r in registers) for id_ in arm.ids})
        return move() or ""

    monkeypatch.setattr("builtins.input", hold_the_pose)
    monkeypatch.setattr("lerobot.motors.motors_bus.enter_pressed", move)

    calibration = arm.record_calibration(
        "position", full_turn_motors=["dummy_2"], inverted_motors=["dummy_3"]
    )

    assert at_prompt == {1: (0, 3, 0), 2: (0, 3, 0), 3: (0, 3, 1)}
    assert calibration == {
        "dummy_1": MotorCalibration(id=1, drive_mode=0, homing_offset=-89, range_min=100, range_max=900),
        "dummy_2": MotorCalibration(id=2, drive_mode=0, homing_offset=211, range_min=0, range_max=1023),
        "dummy_3": MotorCalibration(id=3, drive_mode=1, homing_offset=111, range_min=200, range_max=700),
    }


@pytest.mark.parametrize(
    "selection",
    [
        {"motors": ["dummy_4"]},
        {"full_turn_motors": ["dummy_4"]},
        {"full_turn_motors": ["dummy_3"], "motors": ["dummy_1", "dummy_2"]},
        {"inverted_motors": ["dummy_3"], "motors": ["dummy_1", "dummy_2"]},
    ],
    ids=[
        "unknown motor",
        "unknown full-turn motor",
        "full-turn motor not calibrated",
        "inverted motor not calibrated",
    ],
)
def test_record_calibration_refuses_a_selection_before_touching_the_motors(selection, arm, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("the pose was asked"))

    with pytest.raises(ValueError, match="dummy_"):
        arm.record_calibration("position", **{"full_turn_motors": [], **selection})

    assert (arm._bus.torques, arm._bus.writes) == ([], [])


type _Stops = tuple[int, int] | Callable[["_Joints"], tuple[int, int]]


class _Joints:
    """Physics for the `arm`. Each joint has an angle, in encoder ticks, which its encoder reads
    shifted by the homing offset and wrapped around, as a Dynamixel-like dummy motor does. Each
    `time.sleep` is a tick in which every joint turns towards its goal, at most `speed` ticks,
    but no further than its stops. Stops may be a function of the joints, so that they depend
    on a joint's torque limit or another joint's angle. A joint with a `sag` settles that many
    ticks below its goal, as one holding a load against gravity. Each joint reports the load its
    servo applies, in 0.1% of its maximum torque, as an STS3215 does: what it `needs` to turn at
    its angle (nothing by default), plus 4 for each tick its goal leads it, up to its torque
    limit."""

    def __init__(
        self,
        arm: MockMotorsBus,
        stops: dict[int, _Stops],
        start: int = 2047,
        speed: int = 10_000,
        sag: dict[int, int | Callable[[int], int]] | None = None,
        needs: dict[int, Callable[[int], int]] | None = None,
        speeds: dict[int, Callable[["_Joints"], int]] | None = None,
    ):
        self.bus, self.stops, self.speed, self.sag = arm._bus, stops, speed, sag or {}
        self.needs = needs or {}
        self.speeds = speeds or {}
        self.ticks = 0
        self.angles = dict.fromkeys(arm.ids, start)
        self.resolution = arm.resolution("model_4")
        for id_ in arm.ids:
            self.bus.seed(id_, "goal_position", start)
        self._encode()
        for name in ("read_register_with_error", "sync_read_register_with_error"):
            read = getattr(self.bus, name)
            setattr(
                self.bus, name, lambda *args, read=read, **kwargs: (self._encode(), read(*args, **kwargs))[1]
            )

    def _encode(self) -> None:
        for id_, angle in self.angles.items():
            offset = self.bus.stored(id_, "homing_offset")
            reading = (angle + offset) % self.resolution
            self.bus.seed(id_, "present_position", reading)
            lead = abs(self.bus.stored(id_, "goal_position") - reading)
            need = self.needs.get(id_, lambda angle: 0)(angle)
            limit = self.bus.registers.get((id_, "torque_limit"), 1000)
            self.bus.seed(id_, "present_load", min(limit, need + 4 * lead))

    def tick(self, _seconds: float) -> None:
        self.ticks += 1
        for id_, stops in self.stops.items():
            low, high = stops(self) if callable(stops) else stops
            angle = self.angles[id_]
            target = self.bus.stored(id_, "goal_position") - self.bus.stored(id_, "homing_offset")
            # The goal is an encoder reading: the joint turns the shorter way to the angle it reads.
            sag = self.sag.get(id_, 0)
            sag = sag(angle) if callable(sag) else sag
            target += round((angle - target) / self.resolution) * self.resolution - sag
            speed = self.speeds[id_](self) if id_ in self.speeds else self.speed
            turned = angle + max(-speed, min(speed, target - angle))
            self.angles[id_] = max(low, min(high, turned))
        self._encode()


@pytest.fixture
def loaded_arm(arm, monkeypatch):
    """The `arm`, its motors with a `Torque_Limit` at its maximum, and reporting their load."""
    monkeypatch.setattr(DUMMY_3, "_registers", DUMMY_3._registers | {"torque_limit", "present_load"})
    for id_ in arm.ids:
        arm._bus.seed(id_, "torque_limit", 1000)
    return arm


def _limits(arm, motor_id: int) -> list[int]:
    """The torque limits written to a motor for its pushes, its own limit restored after each left out."""
    return [
        v
        for id_, register, v in arm._bus.writes
        if register == "torque_limit" and id_ == motor_id and v != 1000
    ]


@pytest.fixture
def limited_arm(arm, monkeypatch):
    """The `arm`, its motors with a `Torque_Limit` at its maximum."""
    monkeypatch.setattr(DUMMY_3, "_registers", DUMMY_3._registers | {"torque_limit"})
    for id_ in arm.ids:
        arm._bus.seed(id_, "torque_limit", 1000)
    return arm


def test_find_hard_stop_returns_where_the_joint_stalls_and_stops_pushing(arm, monkeypatch):
    joints = _Joints(arm, {1: (1500, 2600)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    assert arm.find_hard_stop("dummy_1", 1) == 2600
    assert arm.find_hard_stop("dummy_1", -1) == 1500
    assert arm._bus.stored(1, "goal_position") == 1500


def test_find_hard_stop_waits_for_a_slow_joint_that_still_moves(arm, monkeypatch):
    """A joint trailing its goal by more than `stall_error` is not stalled while it moves."""
    joints = _Joints(arm, {1: (1500, 2600)}, speed=5)
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    assert arm.find_hard_stop("dummy_1", 1, step=8, stall_error=40) == 2600


def test_find_hard_stop_frees_a_joint_that_falls_past_the_end_of_its_encoder(arm, monkeypatch):
    """Sagging 100 ticks below its goal, the joint reads across the end of its encoder before its
    goal reaches it: its torque goes off rather than its servo turning the long way back."""
    joints = _Joints(arm, {1: (-500, 4095)}, start=150, sag={1: 100})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)
    arm.enable_torque()

    with pytest.raises(RuntimeError, match="fell past the end of its encoder"):
        arm.find_hard_stop("dummy_1", -1)

    assert arm._bus.stored(1, "torque_enable") == 0


def test_find_hard_stop_refuses_to_run_past_the_end_of_the_encoder(arm, monkeypatch):
    joints = _Joints(arm, {1: (0, 4095)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    with pytest.raises(RuntimeError, match="no hard stop"):
        arm.find_hard_stop("dummy_1", 1)


def test_record_calibration_from_hard_stops(limited_arm, monkeypatch):
    """Joints are searched in the order given, each push at a reduced torque, returned to where
    they started, and freed; ranges sit `margin` inside the stops, a full-turn motor spans its
    encoder."""
    arm = limited_arm
    joints = _Joints(arm, {1: (1000, 3000), 2: (0, 4095), 3: (1800, 2400)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops(
        "position",
        full_turn_motors=["dummy_2"],
        inverted_motors=["dummy_3"],
        motors=["dummy_3", "dummy_1", "dummy_2"],
        torque_limits=[200],
        margin=20,
    )

    assert calibration == {
        "dummy_1": MotorCalibration(id=1, drive_mode=0, homing_offset=47, range_min=1067, range_max=3027),
        "dummy_2": MotorCalibration(id=2, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
        "dummy_3": MotorCalibration(id=3, drive_mode=1, homing_offset=-53, range_min=1767, range_max=2327),
    }
    limits = [(id_, value) for id_, register, value in arm._bus.writes if register == "torque_limit"]
    assert limits == [(3, 200), (3, 1000)] * 2 + [(1, 200), (1, 1000)] * 2
    assert joints.angles == {1: 2047, 2: 2047, 3: 2047}
    assert arm._bus.torques[-1][:2] == ([3, 1, 2], False)


def test_a_joint_too_weak_at_low_torque_is_pushed_harder_until_its_stop_holds(
    limited_arm, monkeypatch, caplog
):
    """Lifting against gravity, the joint stalls short of its stop below a torque of 300."""
    arm = limited_arm

    def lifting(joints: _Joints) -> tuple[int, int]:
        return 1000, (3000 if joints.bus.stored(1, "torque_limit") >= 300 else 2300)

    joints = _Joints(arm, {1: lifting, 2: (0, 4095), 3: (0, 4095)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops(
        "position", full_turn_motors=["dummy_2", "dummy_3"], torque_limits=[150, 300, 500]
    )

    assert (calibration["dummy_1"].range_min, calibration["dummy_1"].range_max) == (1067, 3027)
    assert "may be short" not in caplog.text


def test_a_stop_still_moving_at_the_strongest_push_is_reported(limited_arm, monkeypatch, caplog):
    arm = limited_arm

    def giving_way(joints: _Joints) -> tuple[int, int]:
        return 1000, 2000 + joints.bus.stored(1, "torque_limit")

    joints = _Joints(arm, {1: giving_way, 2: (0, 4095), 3: (0, 4095)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops(
        "position", full_turn_motors=["dummy_2", "dummy_3"], torque_limits=[150, 300, 500]
    )

    assert calibration["dummy_1"].range_max == 2777
    assert "may be short" in caplog.text
    assert "dummy_1 high" in caplog.text


def test_a_stop_hidden_by_an_obstacle_is_reached_from_another_posture(arm, monkeypatch):
    """`dummy_1` meets an obstacle before its low stop unless `dummy_3` is near its low end, and
    before its high stop unless `dummy_3` is near its high end: no single posture reaches both."""

    def obstructed(joints: _Joints) -> tuple[int, int]:
        other = joints.angles[3]
        return (1000 if other < 1900 else 1600), (3000 if other > 2300 else 2500)

    joints = _Joints(arm, {1: obstructed, 2: (0, 4095), 3: (1800, 2400)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops(
        "position",
        full_turn_motors=["dummy_2"],
        motors=["dummy_3", "dummy_1", "dummy_2"],
        postures={"dummy_1": [{}, {"dummy_3": 0.0}, {"dummy_3": 1.0}]},
    )

    assert (calibration["dummy_1"].range_min, calibration["dummy_1"].range_max) == (1067, 3027)
    assert joints.angles == {1: 2047, 2: 2047, 3: 2047}


@pytest.mark.parametrize(
    "stops, calibrated",
    [((1900, 4300), (-1053, 867, 3227)), ((-300, 2400), (997, 717, 3377))],
    ids=["high stop past half a turn", "low stop past half a turn"],
)
def test_a_stop_more_than_half_a_turn_from_the_start_is_reached(stops, calibrated, arm, monkeypatch):
    """The joint starts far nearer one stop than the other: its encoder is re-homed on the way,
    and its range ends up centred on the encoder, its stops `margin` inside."""
    joints = _Joints(arm, {1: stops, 2: (0, 4095), 3: (0, 4095)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops("position", full_turn_motors=["dummy_2", "dummy_3"])

    cal = calibration["dummy_1"]
    assert (cal.homing_offset, cal.range_min, cal.range_max) == calibrated
    assert joints.angles[1] == 2047
    assert arm._bus.torques[-1][:2] == ([1, 2, 3], False)


def test_a_joint_without_stops_within_a_turn_is_refused(arm, monkeypatch):
    joints = _Joints(arm, {1: (-10_000, 10_000), 2: (0, 4095), 3: (0, 4095)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    with pytest.raises(RuntimeError, match="dummy_1"):
        arm.record_calibration_from_hard_stops("position", full_turn_motors=["dummy_2", "dummy_3"])

    assert arm._bus.torques[-1][:2] == ([1, 2, 3], False)


def test_move_to_position_accepts_a_joint_settled_within_tolerance(arm, monkeypatch):
    """Sagging 30 ticks under its load, the joint is close enough at a tolerance of 80 only."""
    joints = _Joints(arm, {1: (0, 4095)}, sag={1: 30})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    with pytest.raises(RuntimeError, match="30 ticks away"):
        arm.move_to_position("dummy_1", 1200, tolerance=20)
    assert arm.move_to_position("dummy_1", 1200, tolerance=80) == 1170


def test_record_calibration_from_hard_stops_tolerates_a_sagging_joint(arm, monkeypatch):
    joints = _Joints(arm, {1: (1000, 3000), 2: (0, 4095), 3: (0, 4095)}, sag={1: 30})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops("position", full_turn_motors=["dummy_2", "dummy_3"])

    assert (calibration["dummy_1"].range_min, calibration["dummy_1"].range_max) == (1067, 3027)


def test_a_follower_keeps_the_hand_clear_of_the_table_while_the_shoulder_tips(arm, monkeypatch):
    """`dummy_1` is a shoulder and `dummy_3` an elbow: the hand meets the table once the forearm
    tips 400 ticks either way, so the shoulder reaches its stops only with the elbow cancelling
    its turn. Following the other way only meets the table sooner, which the union overrides."""

    def shoulder(joints: _Joints) -> tuple[int, int]:
        elbow = joints.angles[3] - 2047
        return max(1000, 2047 - 400 - elbow), min(3000, 2047 + 400 - elbow)

    joints = _Joints(arm, {1: shoulder, 2: (0, 4095), 3: (500, 3600)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops(
        "position",
        full_turn_motors=["dummy_2"],
        motors=["dummy_3", "dummy_1", "dummy_2"],
        postures={"dummy_1": [{}, {"dummy_3": Follow(1.0)}, {"dummy_3": Follow(-1.0)}]},
    )

    assert (calibration["dummy_1"].range_min, calibration["dummy_1"].range_max) == (1067, 3027)
    assert joints.angles == {1: 2047, 2: 2047, 3: 2047}


@pytest.mark.parametrize(
    "sag, ends",
    [(10, {"high"}), (-10, {"low"}), (0, {"low", "high"})],
    ids=["pulled to lower readings", "pulled to higher readings", "no pull: both ends"],
)
def test_the_survey_raises_a_joint_against_gravity(sag, ends, arm, monkeypatch):
    """`dummy_3`, a wrist, is raised while `dummy_1` is searched: at the end of its range gravity
    pulls away from, as the survey felt it sag, or at both ends when it felt none."""
    seen: set[str] = set()

    def shoulder(joints: _Joints) -> tuple[int, int]:
        wrist = joints.angles[3]
        if abs(joints.angles[1] - 2047) > 200:  # the shoulder is being searched
            seen.update({"high"} if wrist > 2500 else {"low"} if wrist < 1600 else set())
        return 1000, 3000

    joints = _Joints(arm, {1: shoulder, 2: (0, 4095), 3: (1500, 2600)}, sag={3: sag})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    arm.record_calibration_from_hard_stops(
        "position",
        full_turn_motors=["dummy_2"],
        motors=["dummy_3", "dummy_1", "dummy_2"],
        postures={"dummy_1": [{"dummy_3": Raised()}]},
    )

    assert seen == ends


def test_load_stall_ends_a_push_as_soon_as_the_joint_presses_its_stop(loaded_arm, monkeypatch):
    """Still at its stop and pushing at its limit, the joint is stalled without its goal first
    running `stall_error` ahead of it."""
    durations = {}
    for load_stall in (None, 0.9):
        joints = _Joints(loaded_arm, {1: (1000, 2600)})
        monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)
        loaded_arm._bus.seed(1, "torque_limit", 150)
        assert loaded_arm.find_hard_stop("dummy_1", 1, stall_error=200, load_stall=load_stall) == 2600
        durations[load_stall] = joints.ticks
    assert durations[0.9] < durations[None] - 15


def test_a_stop_met_with_torque_to_spare_is_pressed_only_once(loaded_arm, monkeypatch):
    """Moving freely up to its stops, the joint is pushed at the weakest torque only."""
    joints = _Joints(loaded_arm, {1: (1000, 3000), 2: (0, 4095), 3: (0, 4095)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = loaded_arm.record_calibration_from_hard_stops(
        "position", full_turn_motors=["dummy_2", "dummy_3"], torque_limits=[150, 300, 500]
    )

    assert (calibration["dummy_1"].range_min, calibration["dummy_1"].range_max) == (1067, 3027)
    assert _limits(loaded_arm, 1) == [150, 150]


def test_a_joint_slowing_as_its_load_nears_the_limit_is_pushed_harder(loaded_arm, monkeypatch, caplog):
    """Lifting against gravity, the joint needs more torque the higher it goes: below a torque of
    300 it slows down to a stall short of its stop, and at 300 it gets there."""
    arm = loaded_arm

    def weak(joints: _Joints) -> bool:
        return joints.bus.stored(1, "torque_limit") < 300

    def lifting(joints: _Joints) -> tuple[int, int]:
        return 1000, (2300 if weak(joints) else 3000)

    def speed(joints: _Joints) -> int:
        return max(0, (2300 - joints.angles[1]) // 25) if weak(joints) else 10_000

    joints = _Joints(
        arm,
        {1: lifting, 2: (0, 4095), 3: (0, 4095)},
        needs={1: lambda a: max(0, (a - 1800) * 3 // 10)},
        speeds={1: speed},
    )
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops(
        "position", full_turn_motors=["dummy_2", "dummy_3"], torque_limits=[150, 300, 500]
    )

    assert (calibration["dummy_1"].range_min, calibration["dummy_1"].range_max) == (1067, 3027)
    assert 300 in _limits(arm, 1)
    assert "may be short" not in caplog.text


def test_a_stop_confirmed_in_one_posture_is_not_pressed_hard_again(limited_arm, monkeypatch):
    """Without a load to tell a stop from a weak joint, the first posture confirms each stop
    with a stronger push; the second only touches it."""
    joints = _Joints(limited_arm, {1: (1000, 3000), 2: (0, 4095), 3: (1500, 2600)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    limited_arm.record_calibration_from_hard_stops(
        "position",
        full_turn_motors=["dummy_2"],
        motors=["dummy_3", "dummy_1", "dummy_2"],
        torque_limits=[150, 500],
        postures={"dummy_1": [{"dummy_3": 0.4}, {"dummy_3": 0.6}]},
    )

    assert _limits(limited_arm, 1).count(500) == 2


def test_the_survey_reads_gravity_from_the_held_sag_without_moving(arm, monkeypatch):
    """Held where it is, the joint settles 30 ticks below its goal: gravity pulls it towards lower
    readings, and no move each way is needed to tell."""
    joints = _Joints(arm, {1: (0, 4095)}, sag={1: 30})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)
    arm.enable_torque()

    def move(motor, target, settle=False):
        return arm.move_to_position(motor, target, tolerance=80, settle=settle)

    assert arm._survey_gravity("dummy_1", 100, 4, move) == -1
    goals = [value for id_, register, value in arm._bus.writes if register == "goal_position" and id_ == 1]
    assert all(abs(goal - 2047) <= 8 for goal in goals)  # held, never turned 100 ticks either way


def test_raising_a_link_upright_turns_it_against_gravity_until_balanced(arm, monkeypatch):
    """Gravity pulls the joint towards lower readings, less the closer it is to 2400, where the
    link stands upright: the raise stops near there, within its 5 steps of 60 ticks."""
    joints = _Joints(arm, {1: (0, 4095)}, start=2100, sag={1: lambda angle: max(0, (2400 - angle) // 8)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)
    arm.enable_torque()

    def move(motor, target, settle=False):
        return arm.move_to_position(motor, target, tolerance=80, settle=settle)

    turned = arm._raise_joint("dummy_1", -1, "upright", move)

    assert 180 <= turned <= 300
    assert joints.angles[1] > 2250


@pytest.mark.parametrize(
    "gravity, signs",
    [
        ({"dummy_1": 1, "dummy_3": 1}, [-1.0]),
        ({"dummy_1": -1, "dummy_3": 1}, [1.0]),
        ({"dummy_1": 1}, [1.0, -1.0]),
    ],
    ids=["both pull up", "opposite pulls", "follower unknown: both"],
)
def test_a_level_follower_turns_against_its_gravity_as_the_searched_joint_turns_with_its_own(gravity, signs):
    postures = [{"dummy_3": Follow(1.0, lift=50, level=True)}]

    resolved = SerialMotorsBus._resolve_raised(postures, gravity, "dummy_1")

    assert [p["dummy_3"] for p in resolved] == [Follow(sign, lift=50) for sign in signs]


def test_record_calibration_from_hard_stops_frees_the_motors_when_a_joint_is_blocked(arm, monkeypatch):
    joints = _Joints(arm, {1: (2040, 2050), 2: (0, 4095), 3: (0, 4095)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    with pytest.raises(RuntimeError, match="dummy_1"):
        arm.record_calibration_from_hard_stops("position", full_turn_motors=["dummy_2", "dummy_3"])

    assert arm._bus.torques[-1][:2] == ([1, 2, 3], False)


@pytest.mark.parametrize(
    "selection",
    [
        {"full_turn_motors": ["dummy_4"]},
        {"postures": {"dummy_2": [{"dummy_1": 0.5}]}},
        {"postures": {"dummy_1": [{"dummy_3": 0.5}]}},
        {"postures": {"dummy_3": [{"dummy_1": 1.5}]}},
    ],
    ids=[
        "unknown motor",
        "posture of a full-turn motor",
        "posture placing a motor searched later",
        "posture outside a range",
    ],
)
def test_record_calibration_from_hard_stops_refuses_a_selection_before_touching_the_motors(selection, arm):
    with pytest.raises(ValueError, match="dummy_"):
        arm.record_calibration_from_hard_stops("position", **{"full_turn_motors": ["dummy_2"], **selection})

    assert (arm._bus.torques, arm._bus.writes) == ([], [])


@pytest.mark.parametrize(
    "motors, order",
    [(None, ["dummy_3", "dummy_2", "dummy_1"]), (["dummy_2", "dummy_1"], ["dummy_2", "dummy_1"])],
    ids=["last motor first by default", "in the order given"],
)
def test_setup_motors_asks_for_one_motor_at_a_time(motors, order, bus, monkeypatch, capsys):
    prompts, set_up = [], []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    monkeypatch.setattr(bus, "setup_motor", set_up.append)

    bus.setup_motors(motors)

    assert set_up == order
    assert prompts == [
        f"Connect the controller board to the '{motor}' motor only and press enter." for motor in order
    ]
    assert f"'{order[0]}' motor id set to {bus.motors[order[0]].id}" in capsys.readouterr().out


def test_a_joint_shoved_by_a_push_is_let_go_before_the_pushed_joint_turns_back(arm, monkeypatch):
    """Near its high stop, `dummy_1` presses the hand into a wall, which shoves `dummy_3` 300 ticks
    from its goal. `dummy_3` is let go where it was shoved before `dummy_1` turns back, so it does
    not spring back into the wall, then put back where it was held."""
    released: list[int] = []

    def wrist(joints: _Joints) -> tuple[int, int]:
        return (2347 if joints.angles[1] > 2800 else 600), 3500

    joints = _Joints(arm, {1: (1000, 3000), 2: (0, 4095), 3: wrist})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)
    write = arm.write

    def watch(register, motor, value, **kwargs):
        if register == "Goal_Position" and motor == "dummy_3" and joints.angles[1] > 2800:
            released.append(value)
        return write(register, motor, value, **kwargs)

    monkeypatch.setattr(arm, "write", watch)

    arm.record_calibration_from_hard_stops(
        "position", full_turn_motors=["dummy_2"], motors=["dummy_1", "dummy_3", "dummy_2"]
    )

    assert released and all(value >= 2340 for value in released)  # let go, shoved, at the wall


def test_a_raise_blocked_at_once_turns_nothing_and_lets_go(arm, monkeypatch):
    """Told gravity pulls towards higher readings, the joint is raised towards lower ones, onto
    what it rests on 10 ticks below: blocked at once, it turns nothing and goes back to its goal."""
    joints = _Joints(arm, {1: (2050, 4095)}, start=2060)
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)
    arm.enable_torque()

    def move(motor, target, settle=False):
        return arm.move_to_position(motor, target, tolerance=80, settle=settle)

    assert arm._raise_joint("dummy_1", 1, "level", move) == 0
    assert arm.read("Goal_Position", "dummy_1", normalize=False) == 2060


def test_a_joint_found_as_wide_as_expected_is_not_searched_in_more_postures(arm, monkeypatch):
    """`dummy_1` reaches both its stops, 2000 ticks apart, in its first posture: with that range
    expected, its second posture, with `dummy_3` turned, is never tried."""
    joints = _Joints(arm, {1: (1000, 3000), 2: (0, 4095), 3: (1000, 3000)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)
    searched_with: list[int] = []
    seek = arm.find_hard_stop

    def watch(motor, *args, **kwargs):
        if motor == "dummy_1":
            searched_with.append(joints.angles[3])
        return seek(motor, *args, **kwargs)

    monkeypatch.setattr(arm, "find_hard_stop", watch)
    arm.record_calibration_from_hard_stops(
        "position",
        full_turn_motors=["dummy_2"],
        motors=["dummy_3", "dummy_1", "dummy_2"],
        postures={"dummy_1": [{}, {"dummy_3": 0.9}]},
        expected_ranges={"dummy_1": 2000},
    )

    assert len(searched_with) == 2  # one push each way, in the first posture only


def test_toward_turns_a_follower_to_its_place_as_the_joint_sweeps(arm, monkeypatch):
    """Gravity pulls `dummy_1` and `dummy_3` towards higher readings. As `dummy_1` sweeps towards
    lower readings, against its gravity, `dummy_3` turns steadily to the place `up` of its range,
    measured from its higher end, gravity's: 0.25 of 1000..3000 is 2500."""
    seen: list[tuple[int, int]] = []

    def stops(joints: _Joints) -> tuple[int, int]:
        seen.append((joints.angles[1], joints.angles[3]))
        return 1000, 3000

    joints = _Joints(arm, {1: stops, 2: (0, 4095), 3: (1000, 3000)}, sag={1: -30, 3: -30})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)
    arm.record_calibration_from_hard_stops(
        "position",
        full_turn_motors=["dummy_2"],
        motors=["dummy_3", "dummy_1", "dummy_2"],
        postures={"dummy_1": [{"dummy_3": Toward(down=0.9, up=0.25, reach=1000)}]},
    )

    at_low_stop = [wrist for shoulder, wrist in seen if shoulder <= 1010]
    assert at_low_stop and all(abs(wrist - 2500) <= 60 for wrist in at_low_stop)
