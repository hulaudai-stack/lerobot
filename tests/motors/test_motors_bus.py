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

from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode, SerialMotorsBus
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


type _Stops = tuple[int, int] | Callable[[MockBus], tuple[int, int]]


class _Joints:
    """Physics for the `arm`: each `time.sleep` is a tick in which every joint moves towards its
    goal, at most `speed` ticks, but no further than its stops. A stop may be a function of the
    whole bus, so that it depends on the joint's torque limit or where another joint is."""

    def __init__(self, arm: MockMotorsBus, stops: dict[int, _Stops], start: int = 2047, speed: int = 10_000):
        self.arm, self.stops, self.speed = arm, stops, speed
        for id_ in arm.ids:
            arm._bus.seed(id_, "present_position", start)
            arm._bus.seed(id_, "goal_position", start)

    def tick(self, _seconds: float) -> None:
        bus = self.arm._bus
        for id_, stops in self.stops.items():
            low, high = stops(bus) if callable(stops) else stops
            present = bus.stored(id_, "present_position")
            goal = bus.stored(id_, "goal_position")
            moved = present + max(-self.speed, min(self.speed, goal - present))
            bus.seed(id_, "present_position", max(low, min(high, moved)))


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
        "dummy_1": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=1020, range_max=2980),
        "dummy_2": MotorCalibration(id=2, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
        "dummy_3": MotorCalibration(id=3, drive_mode=1, homing_offset=0, range_min=1820, range_max=2380),
    }
    limits = [(id_, value) for id_, register, value in arm._bus.writes if register == "torque_limit"]
    assert limits == [(3, 200), (3, 1000)] * 2 + [(1, 200), (1, 1000)] * 2
    assert [arm._bus.stored(id_, "present_position") for id_ in arm.ids] == [2047, 2047, 2047]
    assert arm._bus.torques[-1][:2] == ([3, 1, 2], False)


def test_a_joint_too_weak_at_low_torque_is_pushed_harder_until_its_stop_holds(
    limited_arm, monkeypatch, caplog
):
    """Lifting against gravity, the joint stalls short of its stop below a torque of 300."""
    arm = limited_arm

    def lifting(bus: MockBus) -> tuple[int, int]:
        return 1000, (3000 if bus.stored(1, "torque_limit") >= 300 else 2300)

    joints = _Joints(arm, {1: lifting, 2: (0, 4095), 3: (0, 4095)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops(
        "position", full_turn_motors=["dummy_2", "dummy_3"], torque_limits=[150, 300, 500]
    )

    assert (calibration["dummy_1"].range_min, calibration["dummy_1"].range_max) == (1020, 2980)
    assert "may be short" not in caplog.text


def test_a_stop_still_moving_at_the_strongest_push_is_reported(limited_arm, monkeypatch, caplog):
    arm = limited_arm

    def giving_way(bus: MockBus) -> tuple[int, int]:
        return 1000, 2000 + bus.stored(1, "torque_limit")

    joints = _Joints(arm, {1: giving_way, 2: (0, 4095), 3: (0, 4095)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops(
        "position", full_turn_motors=["dummy_2", "dummy_3"], torque_limits=[150, 300, 500]
    )

    assert calibration["dummy_1"].range_max == 2480
    assert "may be short" in caplog.text
    assert "dummy_1 high" in caplog.text


def test_a_stop_hidden_by_an_obstacle_is_reached_from_another_posture(arm, monkeypatch):
    """`dummy_1` meets an obstacle before its low stop unless `dummy_3` is near its low end, and
    before its high stop unless `dummy_3` is near its high end: no single posture reaches both."""

    def obstructed(bus: MockBus) -> tuple[int, int]:
        other = bus.stored(3, "present_position")
        return (1000 if other < 1900 else 1600), (3000 if other > 2300 else 2500)

    joints = _Joints(arm, {1: obstructed, 2: (0, 4095), 3: (1800, 2400)})
    monkeypatch.setattr("lerobot.motors.motors_bus.time.sleep", joints.tick)

    calibration = arm.record_calibration_from_hard_stops(
        "position",
        full_turn_motors=["dummy_2"],
        motors=["dummy_3", "dummy_1", "dummy_2"],
        postures={"dummy_1": [{}, {"dummy_3": 0.0}, {"dummy_3": 1.0}]},
    )

    assert (calibration["dummy_1"].range_min, calibration["dummy_1"].range_max) == (1020, 2980)
    assert [arm._bus.stored(id_, "present_position") for id_ in arm.ids] == [2047, 2047, 2047]


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
