#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import abc
import logging
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, partial
from pprint import pformat
from typing import TYPE_CHECKING

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _rustypot_available, require_package
from lerobot.utils.utils import enter_pressed, move_cursor_up

if TYPE_CHECKING or _rustypot_available:
    import rustypot
else:
    rustypot = None

# What rustypot raises when the bus does not answer. A motor that answers
# but reports a fault is a different thing, carried by the status error byte.
_TRANSPORT_ERRORS = (RuntimeError, OSError)


@dataclass(frozen=True)
class Follow:
    """In a posture of [`~SerialMotorsBus.record_calibration_from_hard_stops`], a motor that
    follows the one searched: it turns `ratio` ticks for each tick the searched one has turned
    from where it started, within its own range. A ratio of 1 or -1 makes a joint cancel the turn of
    the one searched, e.g. an elbow keeping the forearm level while the shoulder tips. With
    `lift`, it first turns that many ticks against gravity, as the survey found it, and follows
    from there. With `level`, the sign of `ratio` is chosen from the survey so that the follower
    turns against its own gravity as the searched joint turns with its gravity, which keeps a
    forearm level as the shoulder tips; when either pull is unknown, both signs are tried."""

    ratio: float
    lift: int = 0
    level: bool = False


@dataclass(frozen=True)
class Raised:
    """In a posture of [`~SerialMotorsBus.record_calibration_from_hard_stops`], a motor placed
    at the end of its range that gravity pulls away from, as its survey found: a wrist bent
    upward, say. A posture with a motor whose survey found no pull is tried at both ends."""


@dataclass(frozen=True)
class Toward:
    """In a posture of [`~SerialMotorsBus.record_calibration_from_hard_stops`], a calibrated motor
    that turns, as the one searched sweeps from its start towards a stop, steadily from where it is
    to a place in its range, which it reaches once the searched one has turned `reach` ticks, about
    the way to its stop from the middle of its range. Places are fractions of the range from the
    end gravity pulls the motor towards (0) to the end it pulls away from (1), as the survey found:
    `down` while the searched joint turns the way gravity pulls it, `up` while it turns against
    gravity. A forearm folded up as the shoulder tips forward, straightened as it tips back, keeps
    the hand clear of the table in front and of a wall behind. Without a survey pull for either
    motor, the follower stays where it is."""

    down: float
    up: float
    reach: int = 1100


@dataclass(frozen=True)
class Turn:
    """In a posture of [`~SerialMotorsBus.record_calibration_from_hard_stops`], a motor that turns,
    as the one searched sweeps from its start towards a stop, steadily by a number of ticks against
    gravity, as the survey found it (negative: with gravity), from where the arm was put, even if raised
    since, which it has turned once the searched
    one has turned `reach` ticks: `down` while the searched joint turns the way gravity pulls it,
    `up` while it turns against gravity. Unlike [`Toward`], it needs no calibration of the motor,
    only its pull, so it moves the joints searched later out of the way: the upper arm tipped back a
    little while the wrist bends down, say. Without a survey pull for either motor, it stays put."""

    down: int
    up: int
    reach: int = 1000


class EncoderEndError(RuntimeError):
    """A joint driven towards a stop reached the end of its encoder first, at `position`."""

    def __init__(self, message: str, position: int):
        super().__init__(message)
        self.position = position


type NameOrID = str | int
type Value = int | float

logger = logging.getLogger(__name__)


class MotorsBusBase(abc.ABC):
    """
    Base class for all motor bus implementations.

    This is a minimal interface that all motor buses must implement, regardless of their
    communication protocol (serial, CAN, etc.).
    """

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        self.port = port
        self.motors = motors
        self.calibration = calibration if calibration else {}

    @abc.abstractmethod
    def connect(self, handshake: bool = True) -> None:
        """Establish connection to the motors."""
        pass

    @abc.abstractmethod
    def disconnect(self, disable_torque: bool = True) -> None:
        """Disconnect from the motors."""
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the motors."""
        pass

    @abc.abstractmethod
    def read(self, data_name: str, motor: str) -> Value:
        """Read a value from a single motor."""
        pass

    @abc.abstractmethod
    def write(self, data_name: str, motor: str, value: Value) -> None:
        """Write a value to a single motor."""
        pass

    @abc.abstractmethod
    def sync_read(self, data_name: str, motors: str | list[str] | None = None) -> dict[str, Value]:
        """Read a value from multiple motors."""
        pass

    @abc.abstractmethod
    def sync_write(self, data_name: str, values: dict[str, Value]) -> None:
        """Write values to multiple motors."""
        pass

    @abc.abstractmethod
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors."""
        pass

    @abc.abstractmethod
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors."""
        pass

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration parameters from the motors."""
        pass

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Write calibration parameters to the motors."""
        pass


class MotorNormMode(str, Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"


class DriveMode(Enum):
    """Value of a Dynamixel `Drive_Mode` register, and of `MotorCalibration.drive_mode`."""

    NON_INVERTED = 0
    INVERTED = 1


@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode
    motor_type_str: str | None = None
    recv_id: int | None = None


class SerialMotorsBus(MotorsBusBase):
    """Read and write a chain of servos daisy-chained on one serial port, Feetech or Dynamixel.

    rustypot handles each model's metadata (control table, resolution, baud rates, operating
    modes) and the serial I/O; LeRobot handles normalisation, calibration and the recommended
    settings. Motors of several models, such as STS and SCS or XL430 and XL330, share the port.

    To find the port, run:
    ```bash
    lerobot-find-port
    >>> Finding all available ports for the MotorsBus.
    >>> ["/dev/tty.usbmodem575E0032081", "/dev/tty.usbmodem575E0031751"]
    >>> Remove the usb cable from your MotorsBus and press Enter when done.
    >>> The port of this MotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```

    Example for a single Feetech sts3215 on the bus:
    ```python
    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    bus = SerialMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={"my_motor": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)},
    )
    bus.connect()

    position = bus.read("Present_Position", "my_motor", normalize=False)

    # Move a few motor steps as an example
    bus.write("Goal_Position", "my_motor", position + 30, normalize=False)

    bus.disconnect()
    ```
    """

    default_baudrate: int = 1_000_000
    # Milliseconds a status packet is waited for, per attempt: longer than the 16 ms latency
    # timer of a USB adapter. Each retry waits again.
    default_timeout: int = 50
    # Register names are matched without case, as rustypot does.
    normalized_data: frozenset[str] = frozenset({"goal_position", "present_position"})

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ) -> None:
        super().__init__(port, motors, calibration)

        models = {m.model for m in self.motors.values()}
        self._models = {model: self._find_model(model) for model in models}
        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict = {m.id: motor for motor, m in self.motors.items()}

        self._definitions: dict[int, rustypot.ServoDefinition] = {
            m.id: self._models[m.model][1] for m in self.motors.values()
        }
        self._bus: rustypot.Bus | None = None  # open while connected

        self._validate_motors()

    def __len__(self):
        return len(self.motors)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors: \n{pformat(self.motors, indent=8, sort_dicts=False)},\n"
            ")',\n"
        )

    @cached_property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    def _id_to_model(self, motor_id: int) -> str:
        return self._id_to_model_dict[motor_id]

    def _id_to_name(self, motor_id: int) -> str:
        return self._id_to_name_dict[motor_id]

    @staticmethod
    def _find_model(model: str) -> tuple[int, rustypot.ServoDefinition]:
        """The model number of `model` and the rustypot definition that covers it."""
        require_package("rustypot", extra="serial-motors")
        if (found := rustypot.find_model(model)) is None:
            raise ValueError(f"Unknown motor model '{model}': rustypot has no servo of that name.")
        return found

    def _open_bus(self) -> rustypot.Bus:
        """Open rustypot's `Bus` on the port at the default baud rate, with each motor's definition."""
        # LeRobot counts timeouts in milliseconds, rustypot in seconds.
        return rustypot.Bus(self.port, self.default_baudrate, self.default_timeout / 1000, self._definitions)

    @property
    def _connected_bus(self) -> rustypot.Bus:
        """The open rustypot `Bus`."""
        if self._bus is None:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )
        return self._bus

    def _definition(self, motor: NameOrID) -> rustypot.ServoDefinition:
        return self._models[self._get_motor_model(motor)][1]

    def _has_register(self, motor: NameOrID, data_name: str) -> bool:
        return self._definition(motor).register(data_name.lower()) is not None

    def resolution(self, model: str) -> int:
        """Encoder steps per turn of a motor model.

        Args:
            model (`str`):
                Model of one of the bus' motors, e.g. `"sts3215"`.

        Returns:
            `int`: The steps per turn, e.g. 4096 on a 12-bit encoder.

        Raises:
            ValueError: If the model does not count encoder steps.
        """
        if (resolution := self._models[model][1].resolution) is None:
            raise ValueError(f"Motor model '{model}' does not count encoder steps.")
        return resolution

    def _model_number(self, model: str) -> int:
        """The model number a motor of `model` answers with."""
        return self._models[model][0]

    def _inverts_in_software(self, motor: str) -> bool:
        """Whether the calibration's drive mode is applied when normalising. A Dynamixel
        inverts itself through its `Drive_Mode` register; a Feetech has none."""
        return not self._has_register(motor, "Drive_Mode")

    def _get_motor_id(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].id
        elif isinstance(motor, int):
            return motor
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motor_model(self, motor: NameOrID) -> str:
        if isinstance(motor, str):
            return self.motors[motor].model
        elif isinstance(motor, int):
            return self._id_to_model_dict[motor]
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motors_list(self, motors: NameOrID | Sequence[NameOrID] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, int):
            return [self._id_to_name(motors)]
        elif isinstance(motors, Sequence):
            return [m if isinstance(m, str) else self._id_to_name(m) for m in motors]
        else:
            raise TypeError(motors)

    def _get_ids_values_dict(self, values: Value | dict[str, Value] | None) -> dict[int, Value]:
        if isinstance(values, (int | float)):
            return dict.fromkeys(self.ids, values)
        elif isinstance(values, dict):
            return {self.motors[motor].id: val for motor, val in values.items()}
        else:
            raise TypeError(f"'values' is expected to be a single value or a dict. Got {values}")

    def _validate_motors(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"Some motors have the same id!\n{self}")

    def _assert_motors_exist(self) -> None:
        expected_models = {m.id: self._model_number(m.model) for m in self.motors.values()}

        found_models = {}
        for id_ in self.ids:
            model_nb = self.ping(id_)
            if model_nb is not None:
                found_models[id_] = model_nb

        missing_ids = [id_ for id_ in self.ids if id_ not in found_models]
        wrong_models = {
            id_: (expected_models[id_], found)
            for id_, found in found_models.items()
            if found != expected_models[id_]
        }

        if missing_ids or wrong_models:
            error_lines = [f"{self.__class__.__name__} motor check failed on port '{self.port}':"]

            if missing_ids:
                error_lines.append("\nMissing motor IDs:")
                error_lines.extend(
                    f"  - {id_} (expected model: {expected_models[id_]})" for id_ in missing_ids
                )

            if wrong_models:
                error_lines.append("\nMotors with incorrect model numbers:")
                error_lines.extend(
                    f"  - {id_} ({self._id_to_name(id_)}): expected {expected}, found {found}"
                    for id_, (expected, found) in wrong_models.items()
                )

            error_lines.append("\nFull expected motor list (id: model_number):")
            error_lines.append(pformat(expected_models, indent=4, sort_dicts=False))
            error_lines.append("\nFull found motor list (id: model_number):")
            error_lines.append(pformat(found_models, indent=4, sort_dicts=False))

            raise RuntimeError("\n".join(error_lines))

    def _assert_same_firmware(self) -> None:
        """Feetech servos of one definition must run the same firmware: they report it in two
        registers that Dynamixel servos do not have. STS and SCS servos have firmware of their
        own, so each definition is compared on its own."""
        majors = self._sync_read_where("Firmware_Major_Version")
        minors = self._sync_read_where("Firmware_Minor_Version")
        versions: dict[str, dict[str, str]] = {}
        for motor in majors:
            versions.setdefault(self._definition(motor).name, {})[motor] = f"{majors[motor]}.{minors[motor]}"

        if mixed := {name: found for name, found in versions.items() if len(set(found.values())) > 1}:
            raise RuntimeError(
                "Some Motors use different firmware versions:"
                f"\n{pformat(mixed)}\n"
                "Update their firmware first using Feetech's software. "
                "Visit https://www.feetechrc.com/software."
            )

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self._bus is not None

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        """Open the serial port and check the motors.

        Args:
            handshake (`bool`, *optional*, defaults to `True`):
                Whether to check that every motor answers with the model number of its model, and
                that the Feetech motors of one definition run one firmware.

        Raises:
            DeviceAlreadyConnectedError: If the port is already open.
            ConnectionError: If the port could not be opened.
            RuntimeError: If the handshake fails; the port is closed again.
        """

        self._connect(handshake)
        logger.debug(f"{self.__class__.__name__} connected.")

    def _connect(self, handshake: bool = True) -> None:
        try:
            self._bus = self._open_bus()
        except OSError as e:
            raise ConnectionError(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from e
        if not handshake:
            return
        try:
            self._handshake()
        except Exception:
            # Never leave the port open behind a failed handshake: the next
            # connect() would find it already taken.
            self._close()
            raise

    def _close(self) -> None:
        self._connected_bus.close()
        self._bus = None

    def _handshake(self) -> None:
        self._assert_motors_exist()
        self._assert_same_firmware()

    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the serial port, by default with the torque of every motor off.

        Safe to call on a bus that is already disconnected, and the port is released even if
        disabling the torque fails.

        Args:
            disable_torque (`bool`, *optional*, defaults to `True`):
                Whether to disable the torque of every motor before closing the port, so that no
                motor is left resisting.
        """

        if not self.is_connected:
            return

        try:
            if disable_torque:
                self.disable_torque(num_retry=5)
        finally:
            self._close()

        logger.debug(f"{self.__class__.__name__} disconnected.")

    def _scan(self, definition: rustypot.ServoDefinition) -> dict[int, int]:
        """Every id answering at the current baud rate, with its model number: one broadcast
        ping when the servo answers one, which covers a USB adapter's latency once, else one
        read per id."""
        if definition.supports_broadcast_ping:
            return self._connected_bus.broadcast_scan(definition)
        return self._connected_bus.scan(definition)

    def setup_motor(
        self, motor: str, initial_baudrate: int | None = None, initial_id: int | None = None
    ) -> None:
        """Give a single motor its id and the bus' default baud rate.

        The motor must be the only one on the bus. Unless both `initial_baudrate` and
        `initial_id` are given, it is found first: at `initial_baudrate` when only that is
        given, else across the baud rates of its model; an `initial_id` given alone is replaced
        by the id found. rustypot then turns its torque off, opens its EEPROM, and writes the id
        and the baud rate.

        Args:
            motor (`str`):
                Name of the motor.
            initial_baudrate (`int`, *optional*):
                Baud rate the motor answers at now.
            initial_id (`int`, *optional*):
                Id the motor answers at now, used with `initial_baudrate`.

        Raises:
            RuntimeError: If the motor is not found, is not of its model, or does not answer as
                configured once set up.
            ConnectionError: If communication with the motor fails.
        """
        if not self.is_connected:
            self._connect(handshake=False)

        definition = self._definition(motor)
        target_id = self.motors[motor].id
        # The search walks the baud rates too: the port goes back to the default whatever happens.
        try:
            if initial_baudrate is None:
                initial_baudrate, initial_id = self._find_single_motor(motor)

            if initial_id is None:
                _, initial_id = self._find_single_motor(motor, initial_baudrate)

            self.set_baudrate(initial_baudrate)
            try:
                # The motor may answer at an id the bus does not have, or has for another
                # motor: rustypot reaches it there through its definition.
                self._connected_bus.change_id(definition, initial_id, target_id)
                self._connected_bus.change_baudrate(definition, target_id, self.default_baudrate)
            except _TRANSPORT_ERRORS as e:
                raise ConnectionError(
                    f"Failed to set up '{motor}' (motor ID {initial_id} at {initial_baudrate} baud). {e}"
                ) from e
        finally:
            self.set_baudrate(self.default_baudrate)

        # A motor answers the id and baud rate writes before they take effect: check it now
        # answers as configured.
        expected = self._model_number(self.motors[motor].model)
        if (found := self.ping(target_id)) != expected:
            raise RuntimeError(
                f"'{motor}' does not answer as motor ID {target_id} at {self.default_baudrate} baud with "
                f"model number {expected} after its setup (got {found})."
            )

    def setup_motors(self, motors: Sequence[str] | None = None) -> None:
        """Set up the motors one at a time, asking for each to be alone on the bus.

        Args:
            motors (`Sequence[str]`, *optional*):
                Motors in the order they are set up. `None` goes from the last motor to the first,
                from the gripper down on an arm.
        """
        for motor in reversed(self.motors) if motors is None else motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.motors[motor].id}")

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        model = self.motors[motor].model
        definition = self._definition(motor)
        if initial_baudrate is not None:
            search_baudrates = [initial_baudrate]
        else:
            # The default first, so a motor already set up answers on the first try, then
            # the factory rate a new one answers at.
            search_baudrates = sorted(
                definition.baudrates,
                key=lambda rate: (rate != self.default_baudrate, rate != definition.factory_baudrate, rate),
            )
        expected_model_nb = self._model_number(model)

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            id_model = self._scan(definition)
            if id_model:
                found_id, found_model = next(iter(id_model.items()))
                if found_model != expected_model_nb:
                    raise RuntimeError(
                        f"Found a motor at {baudrate} baud with motor ID {found_id}, but its model number "
                        f"{found_model} is different than the one expected: {expected_model_nb}. Make sure "
                        f"only the intended motor, '{motor}' (model '{model}'), is connected."
                    )
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def configure_motors(
        self, return_delay_time: int = 0, maximum_acceleration: int = 254, acceleration: int = 254
    ) -> None:
        """Write LeRobot's recommended settings to every motor.

        Args:
            return_delay_time (`int`, *optional*, defaults to `0`):
                Delay before a motor answers, in units of 2 µs; the motors leave the factory at
                250 (500 µs).
            maximum_acceleration (`int`, *optional*, defaults to `254`):
                Acceleration ceiling, on the servos that have one (Feetech STS), high to speed up
                acceleration and deceleration.
            acceleration (`int`, *optional*, defaults to `254`):
                Acceleration, on the servos that have it (Feetech).
        """
        for motor in self.motors:
            self.write("Return_Delay_Time", motor, return_delay_time)
            if self._has_register(motor, "Maximum_Acceleration"):
                self.write("Maximum_Acceleration", motor, maximum_acceleration)
            if self._has_register(motor, "Acceleration"):
                self.write("Acceleration", motor, acceleration)

            # STS3215: clear Phase bit 4 to keep positions within [0, resolution - 1].
            if self.motors[motor].model == "sts3215":
                phase = self.read("Phase", motor, normalize=False)
                if phase & 0x10:
                    self.write("Phase", motor, phase & ~0x10)

    def set_operating_mode(self, mode: str, motors: NameOrID | Sequence[NameOrID] | None = None) -> None:
        """Put the selected motors in an operating mode, given by name.

        The modes come from each motor's definition, so they depend on the model, not only on
        the family: the Feetech SCS0009 has none. The mode must exist for every selected motor,
        which is checked before any motor changes mode. Write it with the torque off.

        - `position`: holds a goal position within one turn and the position limits.
        - `velocity`: turns endlessly at a goal velocity, as a wheel.
        - `pwm`: drives the motor voltage directly, open loop.
        - `step` (Feetech): moves by the number of steps written as the goal position.
        - `current` (Dynamixel): holds a goal current, that is a torque, whatever the position.
        - `extended_position` (Dynamixel): holds a goal position over several turns. The
          `Min_Position_Limit` and `Max_Position_Limit` registers are ignored in this mode.
        - `current_based_position` (Dynamixel): holds a goal position over several turns with
          the current capped by the goal current, as a gripper that must not crush what it holds.

        Args:
            mode (`str`):
                Name of the operating mode, e.g. `"position"`.
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to set, by name or id. `None` selects every motor.

        Raises:
            ValueError: If a selected motor has no such mode; no motor has changed mode then.
        """
        motor_names = self._get_motors_list(motors)
        for motor in motor_names:
            modes = self._definition(motor).operating_modes
            if mode not in modes:
                raise ValueError(
                    f"Motor '{motor}' ({self.motors[motor].model}) has no '{mode}' operating mode: {sorted(modes)}."
                )
        for motor in motor_names:
            self.write("Operating_Mode", motor, self._definition(motor).operating_modes[mode])

    def _set_torque(
        self, motors: NameOrID | Sequence[NameOrID] | None, enabled: bool, num_retry: int
    ) -> None:
        names = self._get_motors_list(motors)
        ids = [self._get_motor_id(motor) for motor in names]
        failed = self._connected_bus.set_torque(ids, enabled, retries=num_retry)
        if failed:
            details = ", ".join(
                f"'{self._id_to_name(id_)}' (motor ID {id_}): {error}" for id_, error in failed.items()
            )
            raise ConnectionError(
                f"Failed to {'enable' if enabled else 'disable'} torque after {num_retry + 1} tries on {details}."
            )

    @check_if_not_connected
    def disable_torque(self, motors: NameOrID | Sequence[NameOrID] | None = None, num_retry: int = 0) -> None:
        """Disable the torque of the selected motors.

        With the torque off, a motor's EEPROM can be written, and a Feetech motor's lock opens
        for it. Every selected motor is attempted; failures are reported, and unresponsive
        motors may remain torque-enabled.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to free, by name or id. `None` selects every motor.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts on each motor after a communication failure.

        Raises:
            ConnectionError: If some motors could not be reached; the message names them.
        """
        self._set_torque(motors, False, num_retry)

    @check_if_not_connected
    def enable_torque(self, motors: NameOrID | Sequence[NameOrID] | None = None, num_retry: int = 0) -> None:
        """Enable the torque of the selected motors, and close the lock of Feetech motors.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to hold, by name or id. `None` selects every motor.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts on each motor after a communication failure.

        Raises:
            ConnectionError: If some motors could not be reached; the message names them.
        """
        self._set_torque(motors, True, num_retry)

    @contextmanager
    def torque_disabled(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> Iterator[None]:
        """Disable the torque of the selected motors for the duration of a `with` block.

        The torque is enabled again on exit, even when the block raises, which suits configuring
        motors.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to free, by name or id. `None` selects every motor.

        Example:
            ```python
            >>> with bus.torque_disabled():  # doctest: +SKIP
            ...     bus.set_operating_mode("position")
            ```
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    @check_if_not_connected
    def set_baudrate(self, baudrate: int) -> None:
        """Change the baud rate of the open port.

        The port opens at `default_baudrate` on every connect.

        Args:
            baudrate (`int`):
                Baud rate, in bits per second.
        """
        self._connected_bus.set_baudrate(baudrate)

    def _same_calibration(self, motor: str, actual: MotorCalibration) -> bool:
        """Whether the `actual` calibration read from the motor matches the cached one of
        `motor`. The homing offset and the drive mode only count on motors with those
        registers: a Feetech's drive mode lives in the calibration file alone."""
        cached = self.calibration[motor]
        return (
            (cached.range_min, cached.range_max) == (actual.range_min, actual.range_max)
            and (
                cached.homing_offset == actual.homing_offset or not self._has_register(motor, "Homing_Offset")
            )
            and (cached.drive_mode == actual.drive_mode or not self._has_register(motor, "Drive_Mode"))
        )

    @property
    def is_calibrated(self) -> bool:
        """Whether the cached calibration matches the one in the motors.

        Calibrations are matched by motor name: the ids they hold are not compared. The homing
        offset and the drive mode count only on the motors with those registers.
        """
        motors_calibration = self.read_calibration()
        if set(motors_calibration) != set(self.calibration):
            return False
        return all(self._same_calibration(motor, cal) for motor, cal in motors_calibration.items())

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read the calibration of every motor.

        A motor without a homing offset register (Feetech SCS) reports 0, and one without a
        `Drive_Mode` register (Feetech) reports drive mode 0.

        Returns:
            `dict[str, MotorCalibration]`: The calibration of each motor, by name.
        """
        mins = self.sync_read("Min_Position_Limit", normalize=False)
        maxes = self.sync_read("Max_Position_Limit", normalize=False)
        offsets = self._sync_read_where("Homing_Offset")
        drive_modes = self._sync_read_where("Drive_Mode")

        return {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=int(drive_modes.get(motor, 0)),
                homing_offset=int(offsets.get(motor, 0)),
                range_min=int(mins[motor]),
                range_max=int(maxes[motor]),
            )
            for motor, m in self.motors.items()
        }

    def _sync_read_where(self, data_name: str) -> dict[str, Value]:
        """`data_name` of the motors that have the register."""
        motors = [motor for motor in self.motors if self._has_register(motor, data_name)]
        return self.sync_read(data_name, motors, normalize=False) if motors else {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Write a calibration to the motors, and by default cache it.

        The position limits are written to every motor, and the homing offset to the motors
        that have one. Neither the id nor the drive mode is written: a Feetech's drive mode is
        applied in software when normalising, and a Dynamixel's `Drive_Mode` register is set on
        its own.

        Args:
            calibration_dict (`dict[str, MotorCalibration]`):
                Calibration of each motor, by name, as [`~SerialMotorsBus.read_calibration`]
                returns it.
            cache (`bool`, *optional*, defaults to `True`):
                Whether to keep the calibration as the bus' `calibration`, which normalises the
                positions.
        """
        for motor, calibration in calibration_dict.items():
            if self._has_register(motor, "Homing_Offset"):
                self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def reset_calibration(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> None:
        """Restore the factory calibration of the selected motors.

        The homing offset is set to 0 on the motors that have one, and the position limits span
        the whole encoder. The cached `calibration` is cleared.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to reset, by name or id. `None` selects every motor.
        """
        motor_names = self._get_motors_list(motors)

        for motor in motor_names:
            max_res = self.resolution(self._get_motor_model(motor)) - 1
            if self._has_register(motor, "Homing_Offset"):
                self.write("Homing_Offset", motor, 0, normalize=False)
            self.write("Min_Position_Limit", motor, 0, normalize=False)
            self.write("Max_Position_Limit", motor, max_res, normalize=False)

        self.calibration = {}

    def set_half_turn_homings(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> dict[str, int]:
        """Make the present position of each selected motor a half turn.

        The calibration of the motors is reset, then each gets the homing offset that brings its
        present position to half its encoder, e.g. 2047 on a 12-bit one. Every selected motor
        must have a homing offset, which is checked before any register is written.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to home, by name or id. `None` selects every motor.

        Returns:
            `dict[str, int]`: The homing offset written to each motor, by name.

        Raises:
            ValueError: If a selected motor has no homing offset, as the Feetech SCS.
        """
        motor_names = self._get_motors_list(motors)
        signs = {
            motor: sign
            for motor in motor_names
            if (sign := self._definition(motor).homing_offset_sign) is not None
        }
        if without := [motor for motor in motor_names if motor not in signs]:
            raise ValueError(f"Motors {without} have no homing offset.")

        self.reset_calibration(motor_names)
        positions = self.sync_read("Present_Position", motor_names, normalize=False)
        homing_offsets = self._get_half_turn_homings(positions, signs)
        for motor, offset in homing_offsets.items():
            self.write("Homing_Offset", motor, offset)

        return homing_offsets

    def _get_half_turn_homings(
        self, positions: Mapping[str, Value], signs: Mapping[str, int]
    ) -> dict[str, int]:
        """The homing offsets that bring each position to a half turn. The offset adds to the
        position the motor reports on Dynamixel servos and subtracts from it on Feetech ones:
        `present = actual + sign * offset`, the sign coming from the motor's definition.

        A sign-magnitude offset, as on Feetech servos, spans [-2047, 2047] on a 12-bit encoder
        while the half turn of position 4095 needs 2048. That one offset is clamped, which leaves
        the motor one step off the exact half turn instead of failing the write."""
        half_turn_homings = {}
        for motor, position in positions.items():
            offset = signs[motor] * ((self.resolution(self.motors[motor].model) - 1) // 2 - int(position))
            register = self._definition(motor).register("homing_offset")
            if register is None:
                raise ValueError(f"Motor {motor!r} does not support homing offsets")
            if (sign_bit := register.sign_bit) is not None:
                max_magnitude = (1 << sign_bit) - 1
                offset = max(-max_magnitude, min(max_magnitude, offset))
            half_turn_homings[motor] = offset
        return half_turn_homings

    def record_ranges_of_motion(
        self, motors: NameOrID | Sequence[NameOrID] | None = None, display_values: bool = True
    ) -> tuple[dict[str, Value], dict[str, Value]]:
        """Record the range each selected motor is moved through by hand, until ENTER.

        The joints are moved by hand, with the torque off, while a live table shows the
        positions.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to record, by name or id. `None` selects every motor.
            display_values (`bool`, *optional*, defaults to `True`):
                Whether to print the live table.

        Returns:
            `tuple[dict[str, int | float], dict[str, int | float]]`: The lowest and the highest
            raw position of each motor, by name.

        Raises:
            ValueError: If a motor did not move: its lowest and highest positions are equal.
        """
        motor_names = self._get_motors_list(motors)

        start_positions = self.sync_read("Present_Position", motor_names, normalize=False, num_retry=5)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", motor_names, normalize=False, num_retry=5)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motor_names:
                    print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

            if enter_pressed():
                user_pressed_enter = True

            if not user_pressed_enter:
                if display_values:
                    # Move cursor up to overwrite the previous output
                    move_cursor_up(len(motor_names) + 3)
                # Throttle reads even when the live table is disabled.
                time.sleep(0.02)

        same_min_max = [motor for motor in motor_names if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def record_calibration(
        self,
        mode: str,
        full_turn_motors: Sequence[str],
        inverted_motors: Sequence[str] = (),
        motors: Sequence[str] | None = None,
    ) -> dict[str, MotorCalibration]:
        """Calibrate motors moved by hand.

        The selected motors are freed, put in an operating mode and given their direction: a
        motor with a `Drive_Mode` register gets it written, inverted or not, while on the others,
        as Feetech servos, an inversion is applied in software when normalising. The user then
        holds the joints in the middle of their range of motion, which becomes a half turn of
        each motor, and moves each joint through its whole range, recorded until ENTER. A
        full-turn motor is not recorded: it spans the whole encoder of its model.

        The motors need the operating mode and a homing offset, which the Feetech SCS0009 has
        neither. Their homing offsets and position limits are rewritten, the cached
        `calibration` is cleared, and their torque is left off. The calibration returned is not
        written to the position limits nor saved: [`~SerialMotorsBus.write_calibration`] writes
        it, and the device saves it.

        Args:
            mode (`str`):
                Operating mode of the motors, e.g. `"position"`.
            full_turn_motors (`Sequence[str]`):
                Motors given the whole encoder range instead of the range they are moved through.
            inverted_motors (`Sequence[str]`, *optional*, defaults to `()`):
                Motors turning the other way.
            motors (`Sequence[str]`, *optional*):
                Motors to calibrate. `None` calibrates every motor.

        Returns:
            `dict[str, MotorCalibration]`: The calibration of each motor, in the order of the bus.

        Raises:
            ValueError: If a motor is not on the bus, or a full-turn or inverted motor is not one
                of the motors to calibrate; no motor has been touched then.
        """
        motor_names = self._select_calibration_motors(full_turn_motors, inverted_motors, motors)
        drive_modes = self._prepare_calibration(mode, motor_names, inverted_motors)

        input("Move the joints to the middle of their range of motion and press ENTER....")
        homing_offsets = self.set_half_turn_homings(motor_names)

        recorded = [motor for motor in motor_names if motor not in full_turn_motors]
        print(
            f"Move {recorded} sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.record_ranges_of_motion(recorded)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = self.resolution(self.motors[motor].model) - 1

        return {
            motor: MotorCalibration(
                id=self.motors[motor].id,
                drive_mode=drive_modes[motor],
                homing_offset=int(homing_offsets[motor]),
                range_min=int(range_mins[motor]),
                range_max=int(range_maxes[motor]),
            )
            for motor in motor_names
        }

    def _select_calibration_motors(
        self,
        full_turn_motors: Sequence[str],
        inverted_motors: Sequence[str],
        motors: Sequence[str] | None,
    ) -> list[str]:
        """The motors to calibrate, once checked: all on the bus, and every full-turn or
        inverted motor among them. Nothing is written to the motors."""
        motor_names = self._get_motors_list(motors)
        if unknown := [motor for motor in motor_names if motor not in self.motors]:
            raise ValueError(f"Motors {unknown} are not on the bus.")
        if outside := [motor for motor in (*full_turn_motors, *inverted_motors) if motor not in motor_names]:
            raise ValueError(f"Motors {outside} are not among the motors to calibrate: {motor_names}.")
        return motor_names

    def _prepare_calibration(
        self, mode: str, motor_names: list[str], inverted_motors: Sequence[str]
    ) -> dict[str, int]:
        """Free the motors, put them in `mode` and give them their direction. Returns the drive
        mode of each motor, written to those with a `Drive_Mode` register."""
        self.disable_torque(motor_names)
        self.set_operating_mode(mode, motor_names)
        drive_modes = {
            motor: (DriveMode.INVERTED if motor in inverted_motors else DriveMode.NON_INVERTED).value
            for motor in motor_names
        }
        for motor, drive_mode in drive_modes.items():
            if self._has_register(motor, "Drive_Mode"):
                self.write("Drive_Mode", motor, drive_mode)
        return drive_modes

    def move_to_position(
        self,
        motor: str,
        target: int,
        *,
        step: int = 8,
        period_s: float = 0.02,
        tolerance: int = 20,
        settle_cycles: int = 10,
        still_ticks: int = 3,
        settle: bool = False,
        on_step: Callable[[int], None] | None = None,
    ) -> int:
        """Ramp the raw goal of a torque-enabled `motor` to `target`, `step` ticks every
        `period_s`, so the joint never jumps, then wait until it is within `tolerance` of it.

        A joint holding a load settles short of its goal, as a position-controlled servo sags
        under gravity: once it has moved no more than `still_ticks` over `settle_cycles`
        periods, it has gone as far as it will. With `settle`, the joint is always left to settle,
        and the position returned is where it came to rest. `on_step`, if given, is called every
        period with the raw position of the joint, e.g. to move other joints along with it.

        Returns:
            `int`: The raw position reached.

        Raises:
            RuntimeError: If the joint settles further than `tolerance` from the target, e.g. it
                is blocked, or never settles.
        """
        goal = int(self.read("Goal_Position", motor, normalize=False))
        while goal != target:
            goal += max(-step, min(step, target - goal))
            self.write("Goal_Position", motor, goal, normalize=False)
            if on_step is not None:
                on_step(int(self.read("Present_Position", motor, normalize=False)))
            time.sleep(period_s)
        recent: deque[int] = deque(maxlen=settle_cycles + 1)
        for _ in range(20 * settle_cycles):
            position = int(self.read("Present_Position", motor, normalize=False))
            recent.append(position)
            if on_step is not None:
                on_step(position)
            if not settle and abs(position - target) <= tolerance:
                return position
            if len(recent) == recent.maxlen and max(recent) - min(recent) <= still_ticks:
                break
            time.sleep(period_s)
        if abs(position - target) <= tolerance:
            return position
        raise RuntimeError(
            f"Motor '{motor}' did not reach position {target}; it stopped at {position}, "
            f"{abs(position - target)} ticks away. Is something in its way?"
        )

    def find_hard_stop(
        self,
        motor: str,
        direction: int,
        *,
        step: int = 8,
        stall_error: int = 40,
        stall_cycles: int = 5,
        still_ticks: int = 3,
        period_s: float = 0.02,
        guard: int = 0,
        load_stall: float | None = None,
        impact_speed: float = 0.6,
        impact_run: int = 5,
        report: dict[str, int] | None = None,
        on_step: Callable[[int], None] | None = None,
    ) -> int:
        """Drive a torque-enabled `motor` slowly in `direction` (+1 or -1) until it stalls.

        The raw goal moves `step` ticks every `period_s`. Once the joint lags the goal by more
        than `stall_error` ticks the goal stops advancing, which caps how hard the joint pushes.
        The joint is stalled when it lags the goal and has moved no more than `still_ticks` over
        the last `stall_cycles` periods, so a joint that is slow but still moving, as one lifting
        a load, is not mistaken for a stalled one. The goal is then set to the position reached,
        so the motor stops pushing. Limit the motor's torque beforehand: a stall is the joint
        pressing on its stop, or on whatever holds it back at that torque. `on_step`, if given,
        is called every period with the raw position of the joint.

        The goal never comes within `guard` ticks of either end of the encoder, which leaves room
        for a joint that falls past its goal, as one turning with gravity at a low torque does.
        Should its reading still jump across the end of the encoder, its torque is turned off at
        once: a servo told to go back would turn the long way round.

        With `load_stall`, a motor that reports its load (`Present_Load`, in 0.1% of its maximum
        torque) is also stalled as soon as it is still while pushing at least that fraction of
        its torque limit, once it has moved: the push ends as the joint meets its stop, without
        the goal first running `stall_error` ahead of it. A joint still at the start of a push,
        accelerating its link, does not count.

        `report`, if given, receives `impact`: 1 if the joint ran at speed, at least
        `impact_speed` of a step each period for `impact_run` periods, until it stopped within a
        few periods, as a joint meeting something does; 0 if it slowed down to a stall, as a joint
        too weak to lift its load does.

        Returns:
            `int`: The raw position where the joint stalled.

        Raises:
            ValueError: If `direction` is neither +1 nor -1.
            EncoderEndError: If the goal reaches the guard before any stall: the joint has no stop
                within reach, or it would wrap around the encoder.
            RuntimeError: If the reading wrapped around the encoder.
        """
        if direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, not {direction}.")
        max_position = self.resolution(self.motors[motor].model) - 1
        goal = int(self.read("Present_Position", motor, normalize=False))
        recent: deque[int] = deque(maxlen=stall_cycles + 1)
        loaded = load_stall is not None and self._has_register(motor, "Present_Load")
        limit = int(self.read("Torque_Limit", motor, normalize=False)) if loaded else 0
        speeds: deque[int] = deque(maxlen=impact_run + stall_cycles + 6)
        travelled = 0
        if report is not None:
            report.pop("impact", None)
        # The goal advances at most once per period; a joint that keeps creeping without ever
        # catching up is bounded too.
        for _ in range(10 * (max_position // step + stall_cycles)):
            position = int(self.read("Present_Position", motor, normalize=False))
            if recent and abs(position - recent[-1]) > max_position // 2:
                self.disable_torque(motor)
                raise RuntimeError(
                    f"Motor '{motor}' fell past the end of its encoder, from {recent[-1]} to {position}; "
                    "its torque is off."
                )
            moved = direction * (position - recent[-1]) if recent else 0
            travelled = max(travelled, travelled + moved)
            speeds.append(moved)
            recent.append(position)
            if on_step is not None:
                on_step(position)
            load = abs(int(self.read("Present_Load", motor, normalize=False))) if loaded else 0
            lagging = direction * (goal - position) > stall_error
            pressing = (
                loaded and load_stall is not None and travelled >= 3 * step and load >= load_stall * limit
            )
            still = len(recent) == recent.maxlen and max(recent) - min(recent) <= still_ticks
            if (lagging or pressing) and still:
                self.write("Goal_Position", motor, position, normalize=False)
                if report is not None:
                    report["impact"] = int(
                        self._ran_into_something(list(speeds), step, impact_speed, impact_run)
                    )
                return position
            if not lagging:
                goal += direction * step
                if not guard <= goal <= max_position - guard:
                    self.write("Goal_Position", motor, position, normalize=False)
                    raise EncoderEndError(
                        f"Motor '{motor}' found no hard stop before the end of its encoder at {position}.",
                        position,
                    )
                self.write("Goal_Position", motor, goal, normalize=False)
            time.sleep(period_s)
        self.write("Goal_Position", motor, position, normalize=False)
        raise RuntimeError(f"Motor '{motor}' neither stalled nor reached the end of its encoder.")

    @staticmethod
    def _ran_into_something(speeds: list[int], step: int, impact_speed: float, impact_run: int) -> bool:
        """Whether a joint, by the ticks it moved each period up to its stall, ran at speed until it
        stopped abruptly: `impact_run` periods of at least `impact_speed` of a step, then no more
        than a few slower ones."""
        fast = impact_speed * step
        slowing = 0
        while speeds and speeds[-1] < fast:
            speeds.pop()
            slowing += 1
        running = 0
        while speeds and speeds[-1] >= fast:
            speeds.pop()
            running += 1
        return running >= impact_run and slowing <= impact_run

    def _seek_confirmed_stop(
        self,
        motor: str,
        direction: int,
        torque_limits: Sequence[int],
        confirm_tolerance: int,
        seek: Callable[[str, int, dict[str, int]], int],
        trust_impacts: bool = False,
        known: int | None = None,
    ) -> tuple[int, bool]:
        """Seek the stop of `motor` in `direction` at each torque of `torque_limits` in turn,
        each push starting where the last one stalled, until a stronger push gets no further
        than `confirm_tolerance` ticks: then the stall is the stop, not a lack of torque. The
        motor's own torque limit is back after each push.

        A push needs no stronger one after it, and the stall is the stop, when either:
        - with `trust_impacts`, the joint ran at speed until it stopped abruptly: it had torque
          to spare, so something stopped it (see [`~SerialMotorsBus.find_hard_stop`]);
        - the stall is within `confirm_tolerance` of `known`, a stop already confirmed in another
          posture: the same stop, touched again. A stall short of it is pushed harder as any.
        Only a joint that slows as its load nears the limit, as one lifting the arm, is pushed
        harder, so most stops are only ever pressed at the first, weakest torque.

        Returns:
            `tuple[int, bool]`: The furthest position reached, and whether it was confirmed. A
            single push, or none at all without `Torque_Limit`, counts as confirmed.
        """
        limited = bool(torque_limits) and self._has_register(motor, "Torque_Limit")
        limits: list[int | None] = list(torque_limits) if limited else [None]
        own_limit = self.read("Torque_Limit", motor, normalize=False) if limited else None
        furthest: int | None = None
        for limit in limits:
            if limit is not None:
                self.write("Torque_Limit", motor, limit, normalize=False)
            report: dict[str, int] = {}
            try:
                stop = seek(motor, direction, report)
            finally:
                if own_limit is not None:
                    self.write("Torque_Limit", motor, own_limit, normalize=False)
            if furthest is not None and direction * (stop - furthest) <= confirm_tolerance:
                return (stop if direction * (stop - furthest) > 0 else furthest), True
            if (trust_impacts and report.get("impact")) or (
                known is not None and abs(stop - known) <= confirm_tolerance
            ):
                logger.info(f"{motor}: stop at {stop} with torque limit {limit}")
                return stop if furthest is None or direction * (stop - furthest) > 0 else furthest, True
            furthest = stop if furthest is None or direction * (stop - furthest) > 0 else furthest
            logger.info(f"{motor}: stalled at {stop} with torque limit {limit}")
        assert furthest is not None
        return furthest, len(limits) == 1

    def _shift_homing(self, motor: str, delta: int) -> int:
        """Rewrite the homing offset of a torque-enabled `motor` so that it reads `delta` more,
        modulo its encoder, without moving: its torque is off only while the offset is written,
        which a motor's EEPROM needs, and its goal follows the new reading.

        Returns:
            `int`: The shift applied: `delta`, or a step off when a sign-magnitude offset is
            clamped, as in [`~SerialMotorsBus._get_half_turn_homings`].
        """
        definition = self._definition(motor)
        sign, register = definition.homing_offset_sign, definition.register("homing_offset")
        if sign is None or register is None:
            raise ValueError(f"Motor {motor!r} does not support homing offsets")
        resolution = self.resolution(self.motors[motor].model)
        current = int(self.read("Homing_Offset", motor, normalize=False))
        # Offsets a whole turn apart read the same, so the one nearest zero is written.
        offset = (current + sign * delta + resolution // 2) % resolution - resolution // 2
        if (sign_bit := register.sign_bit) is not None:
            max_magnitude = (1 << sign_bit) - 1
            offset = max(-max_magnitude, min(max_magnitude, offset))
        applied = (sign * (offset - current) + resolution // 2) % resolution - resolution // 2

        self.disable_torque(motor)
        try:
            self.write("Homing_Offset", motor, offset, normalize=False)
            position = self.read("Present_Position", motor, normalize=False)
            self.write("Goal_Position", motor, position, normalize=False)
        finally:
            self.enable_torque(motor)
        return applied

    def _survey_gravity(self, motor: str, ticks: int, threshold: int, move: Callable[..., int]) -> int:
        """Turn a held `motor` `ticks` each way from where it is and back, letting it settle each
        time, and tell where gravity pulls it: +1 towards higher readings, -1 towards lower ones,
        0 if it sagged less than `threshold` ticks on average. Friction holds a joint short of
        its goal on either side and cancels out; gravity holds it short the same way on both.
        A side the joint cannot turn to is left out. `move(motor, raw_target, settle=...)`
        moves it and returns where it came to rest."""
        if ticks <= 0:
            return 0
        # Held where it is, a joint settles short of its goal the way gravity pulls it, with no
        # move that could press something, as a hand resting near the table. A sag too small to
        # tell from friction is surveyed by turning the joint each way.
        goal = int(self.read("Goal_Position", motor, normalize=False))
        held = move(motor, goal, settle=True) - goal
        if abs(held) >= 2 * threshold:
            pulled = 1 if held > 0 else -1
            logger.info(
                f"{motor}: held sag {held:+d} ticks, gravity pulls towards {('lower', 'higher')[pulled > 0]} readings"
            )
            return pulled
        start = int(self.read("Present_Position", motor, normalize=False))
        sags = []
        for direction in (1, -1):
            target = start + direction * ticks
            try:
                sags.append(move(motor, target, settle=True) - target)
            except RuntimeError:
                logger.info(
                    f"{motor}: cannot turn {ticks} ticks towards {'higher' if direction > 0 else 'lower'} readings"
                )
            move(motor, start)
        pull = sum(sags) / len(sags) if sags else 0
        pulled = 0 if abs(pull) < threshold else (1 if pull > 0 else -1)
        logger.info(
            f"{motor}: survey sag {pull:+.1f} ticks, gravity pulls {('towards lower readings', 'nowhere clear', 'towards higher readings')[pulled + 1]}"
        )
        return pulled

    def _raise_joint(
        self,
        motor: str,
        pulled: int,
        mode: str,
        move: Callable[..., int],
        step: int = 60,
        max_steps: int = 12,
        threshold: int = 4,
    ) -> int:
        """Turn a held `motor` against gravity, which pulls it towards `pulled` (+1 higher readings,
        -1 lower), `step` ticks at a time, letting it settle after each to measure how far gravity
        still holds it short of its goal, its sag:
        - `"upright"`: until the sag falls to a third of the most seen, or reverses, and no more
          than 5 steps: the link stands over its joint, as an upper arm made vertical, which
          lifts the whole arm;
        - `"level"`: until the sag, having grown, falls back below nine tenths of the most seen:
          the link has passed level and points a little up, as a forearm lifting the hand.
        A joint that cannot turn further, against something or a stop, or turns less than half a
        step, as one pressing down on what it rests on, goes back to where it last turned freely.

        Returns:
            `int`: The ticks the joint turned: none if its first step was blocked, as when the
            survey misread a joint resting on something, which pulls it the other way.
        """
        start = target = int(self.read("Goal_Position", motor, normalize=False))
        try:
            last = move(motor, start, settle=True)
        except RuntimeError:
            last = int(self.read("Present_Position", motor, normalize=False))
        peak = 0.0
        # Upright, a link with others held out past it balances beyond vertical: no further
        # than a few steps, enough to lift a hand off the table, never to tip the arm back.
        for _ in range(max_steps if mode == "level" else min(max_steps, 5)):
            goal = target - pulled * step
            try:
                rest = move(motor, goal, settle=True)
            except RuntimeError:
                rest = None
            if rest is None or -pulled * (rest - last) < step // 2:
                with suppress(RuntimeError):
                    move(motor, target, settle=True)
                break
            load = pulled * (rest - goal)
            peak = max(peak, load)
            target, last = goal, rest
            if mode == "upright" and (load < threshold or load < peak / 3):
                break
            if mode == "level" and peak >= threshold and load < 0.9 * peak:
                break
        logger.info(f"{motor}: raised {abs(target - start)} ticks against gravity ({mode})")
        return abs(target - start)

    @staticmethod
    def _resolve_raised(
        postures: Sequence[Mapping[str, float | Follow | Raised | Toward | Turn]],
        gravity: Mapping[str, int],
        searched: str | None = None,
    ) -> list[dict[str, float | Follow | Toward | Turn]]:
        """Place each [`Raised`] motor of `postures` at the end of its range gravity pulls away
        from, a posture tried at both ends for a motor whose survey found no pull, and give each
        `level` [`Follow`] of the `searched` motor its sign, both signs when a pull is unknown."""
        resolved: list[dict[str, float | Follow | Toward | Turn]] = []
        for posture in postures:
            expanded: list[dict[str, float | Follow | Toward | Turn]] = [{}]
            for other, place in posture.items():
                if isinstance(place, Follow) and place.level:
                    follower, leader = gravity.get(other, 0), gravity.get(searched or "", 0)
                    size = abs(place.ratio)
                    signs = (-follower * leader,) if follower and leader else (1, -1)
                    ends: tuple[float | Follow | Toward | Turn, ...] = tuple(
                        Follow(sign * size, place.lift) for sign in signs
                    )
                elif not isinstance(place, Raised):
                    ends = (place,)
                elif gravity.get(other, 0):
                    ends = (1.0 if gravity[other] < 0 else 0.0,)
                else:
                    ends = (0.0, 1.0)
                expanded = [{**partial, other: end} for partial in expanded for end in ends]
            resolved += expanded
        return resolved

    def record_calibration_from_hard_stops(
        self,
        mode: str,
        full_turn_motors: Sequence[str],
        inverted_motors: Sequence[str] = (),
        motors: Sequence[str] | None = None,
        *,
        torque_limits: Sequence[int] = (),
        postures: Mapping[str, Sequence[Mapping[str, float | Follow | Raised | Toward | Turn]]] | None = None,
        revisit: Mapping[str, Sequence[Mapping[str, float | Follow | Raised | Toward | Turn]]] | None = None,
        lift: Mapping[str, int] | None = None,
        raise_arm: Mapping[str, str] | None = None,
        expected_ranges: Mapping[str, int] | None = None,
        range_tolerance: float = 0.01,
        confirm_tolerance: int = 15,
        return_tolerance: int = 80,
        survey_ticks: int = 100,
        gravity_threshold: int = 4,
        margin: int = 20,
        step: int = 8,
        move_step: int | None = None,
        stall_error: int = 150,
        stall_cycles: int = 3,
        guard: int = 150,
        load_stall: float | None = 0.9,
        trust_impacts: bool = True,
        working_torque_limit: int | None = None,
        move_torque_limit: int | None = None,
        release_ticks: int | None = 50,
        period_s: float = 0.02,
    ) -> dict[str, MotorCalibration]:
        """Calibrate motors by driving each joint into its two hard stops, instead of by hand.

        The motors are prepared as in [`~SerialMotorsBus.record_calibration`], then the pose
        they are in becomes a half turn of each motor. Every motor is then held in place and the
        motors not full-turn are searched one at a time, in the order of `motors`, the low stop
        first. A joint that runs out of encoder before its low stop is re-homed to the middle of
        its encoder and pushed on, and its low stop is then re-homed near the bottom of the
        encoder, so that its high stop fits too: any joint whose stops are less than a turn apart
        is searched, whatever pose it started in. Once both stops are found, the joint is re-homed
        so that the middle of its range is a half turn, as a manual calibration makes it.

        A joint is searched in both directions, once per posture of the joints searched before
        it (`postures`), and its range spans the furthest stops found over all of them: an
        obstacle in one posture, as the table or another link, only makes that trial stop short,
        so a stop met in another posture still counts. In each direction the joint is pushed at
        each torque of `torque_limits` in turn until a stronger push gets no further, which tells
        a stop from a joint too weak to lift its load. Between pushes and trials the joints go
        back to where they started, at their own torque. Each range is the two stops brought
        `margin` ticks inward, so the joints are never commanded into their stops afterwards. A
        full-turn motor is not searched: it spans the whole encoder of its model.

        Like `record_calibration`, the homing offsets and position limits of the motors are
        rewritten, their torque is left off, and the calibration returned is neither written
        to the position limits nor saved.

        Args:
            mode (`str`):
                Operating mode of the motors, e.g. `"position"`.
            full_turn_motors (`Sequence[str]`):
                Motors given the whole encoder range instead of being searched.
            inverted_motors (`Sequence[str]`, *optional*, defaults to `()`):
                Motors turning the other way.
            motors (`Sequence[str]`, *optional*):
                Motors to calibrate, in the order they are searched. `None` calibrates every
                motor in the order of the bus.
            torque_limits (`Sequence[int]`, *optional*, defaults to `()`):
                Raw `Torque_Limit` of a motor for each push towards a stop, weakest first.
                Empty, or a motor without that register, pushes once at the motor's own limit.
            postures (`Mapping[str, Sequence[Mapping[str, float | Follow | Raised]]]`, *optional*):
                For a motor, the postures to search it from. A posture places motors searched
                before it at a fraction of their range, 0 at its low end and 1 at its high end,
                [`Raised`] against gravity, or makes them [`Follow`] it as it turns; the others
                stay where they started. A motor without postures is searched once, as the
                joints started.
            revisit (`Mapping[str, Sequence[Mapping[str, float | Follow | Raised]]]`, *optional*):
                Motors searched again once every motor has been, from these postures, which may
                place any motor searched: a stop that depends on a joint calibrated later, as the
                elbow folding as far as the shoulder's tilt allows. Each range spans the furthest
                stops of every visit.
            lift (`Mapping[str, int]`, *optional*):
                Ticks each of these motors turns against gravity, as the survey found it, before
                any stop is sought, and holds until its own search, after which it goes back to
                where it started: an elbow lifting a hand that started low, clear of the table the
                first searches could press it into. A motor the survey found no pull on stays
                where it is.
            expected_ranges (`Mapping[str, int]`, *optional*):
                Ticks between the two stops of a joint, as its design has them. A posture that finds
                the range this wide, within `range_tolerance`, has found both its stops: it ends the
                search of that joint: no other posture, nor its revisit, is tried. A range found
                more than 5% off is reported.
            range_tolerance (`float`, *optional*, defaults to `0.01`):
                How much narrower than expected a range may be and still count as complete.
            raise_arm (`Mapping[str, str]`, *optional*):
                Motors turned against gravity before anything else, in this order, each surveyed
                just before: `"upright"` until its link stands over it, `"level"` until its link
                points a little above level (see [`~SerialMotorsBus._raise_joint`]). Each holds
                there until its own search, which starts from where the arm was put. A hand
                resting on the table is lifted off it, so that no search presses it there.
            confirm_tolerance (`int`, *optional*, defaults to `15`):
                Ticks a stronger push may gain over the last one and still confirm the stop.
            return_tolerance (`int`, *optional*, defaults to `80`):
                Ticks a joint going back to where it started, or to a posture, may settle away
                from it: a joint holding a load sags short of its goal.
            survey_ticks (`int`, *optional*, defaults to `100`):
                Before any stop is sought, each joint searched is turned this far each way from
                its start and back, at its own torque: where it settles tells the way gravity
                pulls it, as it sags short of its goal on that side. 0 skips the survey.
            gravity_threshold (`int`, *optional*, defaults to `4`):
                Ticks of average sag below which the survey finds no pull.
            margin (`int`, *optional*, defaults to `20`):
                Ticks between each stop and the end of the range recorded.
            move_step (`int`, *optional*):
                Ticks per period of the moves through space the arm has swept already, back to
                where the joints started or into a posture; `None` moves as fast as `step`. The
                pushes towards an unknown stop keep to `step`: faster, a joint meets what stops it
                harder, and can wedge a link against it.
            load_stall (`float`, *optional*, defaults to `0.9`):
                Passed to [`~SerialMotorsBus.find_hard_stop`]: a joint still while pushing this
                fraction of its torque limit has met its stop. `None` waits for `stall_error`.
            working_torque_limit (`int`, *optional*):
                Raw `Torque_Limit` of every motor for the whole calibration, restored after it:
                what holding and moving the arm may use, while pushes towards a stop use
                `torque_limits`. `None` leaves the motors' own limits.
            move_torque_limit (`int`, *optional*):
                Raw `Torque_Limit` of a motor while it moves between searches: a move that meets
                something, as a hand brushing a wall, presses no harder than that. Only a move
                blocked at it is made again at the motor's own limit. `None` moves at the own limit.
            release_ticks (`int`, *optional*, defaults to `50`):
                A joint pushed further than this from its goal while another met a wall, as the
                shoulder when the hand presses on one, is let go where it is before that other joint
                turns back, then put back: holding on, it would press the hand into the wall as it
                springs back, and wedge it. `None` holds every joint throughout.
            trust_impacts (`bool`, *optional*, defaults to `True`):
                A stop is confirmed by its first push when the joint ran into it at speed: no
                stronger push follows. `False` confirms every stop with a stronger push.
            step, stall_error, stall_cycles, guard, period_s:
                Passed to [`~SerialMotorsBus.find_hard_stop`]. `stall_error` is how far the goal may
                lead a joint: a position servo pushes in proportion to it, so it must be large
                enough for the push to reach the torque limit, or a stronger limit pushes no
                harder. A joint never comes within `guard` ticks of the ends of its encoder.

        Returns:
            `dict[str, MotorCalibration]`: The calibration of each motor, in the order of the bus.

        Raises:
            ValueError: If a motor is not on the bus, a full-turn or inverted motor is not one of
                the motors to calibrate, or a posture places a motor not searched before the one
                it is for, or outside its range; no motor has been touched then.
            RuntimeError: If a joint finds no stop, cannot return to its start, or its stops are
                closer than twice `margin`. The torque of every motor is off then.
        """
        motor_names = self._select_calibration_motors(full_turn_motors, inverted_motors, motors)
        searched = [motor for motor in motor_names if motor not in full_turn_motors]
        postures = postures or {}
        for motor, motor_postures in postures.items():
            if motor not in searched:
                raise ValueError(f"Postures are given for '{motor}', which is not searched.")
            before = searched[: searched.index(motor)]
            for posture in motor_postures:
                # A turn moves a joint from where it is, so it may be one searched later.
                if misplaced := [
                    other
                    for other, place in posture.items()
                    if other not in before
                    and not (isinstance(place, Turn) and other in searched and other != motor)
                ]:
                    raise ValueError(f"Postures of '{motor}' place {misplaced}, not searched before it.")
                if outside := [
                    other
                    for other, place in posture.items()
                    if not isinstance(place, (Follow, Raised, Toward, Turn)) and not 0 <= place <= 1
                ]:
                    raise ValueError(f"Postures of '{motor}' place {outside} outside their range.")

        revisit = revisit or {}
        for motor, motor_postures in revisit.items():
            if motor not in searched:
                raise ValueError(f"'{motor}' is revisited but not searched.")
            for posture in motor_postures:
                if misplaced := [other for other in posture if other not in searched or other == motor]:
                    raise ValueError(
                        f"Revisit postures of '{motor}' place {misplaced}, not other motors searched."
                    )

        drive_modes = self._prepare_calibration(mode, motor_names, inverted_motors)
        homing_offsets = self.set_half_turn_homings(motor_names)

        range_mins: dict[str, int] = {}
        range_maxes: dict[str, int] = {}

        # Positions are tracked in each motor's frame at the start, as the homing of a motor may
        # shift while it is searched: what it reads is that position plus `shifts[motor]`.
        shifts = dict.fromkeys(motor_names, 0)
        middles = {motor: (self.resolution(self.motors[motor].model) - 1) // 2 for motor in motor_names}

        def rehome(motor: str, position: int, reading: int) -> None:
            """Shift the homing of `motor` so that `position` reads `reading`."""
            if delta := reading - (position + shifts[motor]):
                shifts[motor] += self._shift_homing(motor, delta)

        # The joints following the one searched in the current posture: their ratio, and where
        # each was when the searched one was at its start.
        followers: dict[str, tuple[float, int]] = {}
        # The joints turning towards a place as the one searched sweeps: where each was at the
        # start, and its goal while the searched one turns towards lower and higher readings.
        towards: dict[str, tuple[int, int, int, int]] = {}

        # How far the searched joint has turned from its start, as its followers see it: a ratchet
        # that only moves in the direction the joint is driven (`motion`), so followers neither
        # lead the joint, as following its goal would when it is held back, nor go back and forth.
        follow_state = {"turned": 0, "motion": 0}

        def follow(motor: str, reading: int) -> None:
            turned = reading - shifts[motor] - starts[motor]
            if follow_state["motion"] * (turned - follow_state["turned"]) <= 0:
                return
            follow_state["turned"] = turned
            for other, (ratio, base) in followers.items():
                target = base + shifts[other] + round(ratio * follow_state["turned"])
                target = max(range_mins[other], min(range_maxes[other], target))
                self.write("Goal_Position", other, target, normalize=False)
            for other, (base, to_lower, to_higher, reach) in towards.items():
                start = base + shifts[other]
                goal = to_higher if follow_state["turned"] > 0 else to_lower
                progress = min(1.0, abs(follow_state["turned"]) / reach)
                self.write("Goal_Position", other, round(start + progress * (goal - start)), normalize=False)

        def on_step(motor: str) -> Callable[[int], None] | None:
            return (lambda reading: follow(motor, reading)) if followers or towards else None

        # The stops found so far of the joint searched: (position, confirmed) by direction.
        stops: dict[int, tuple[int, bool]] = {}

        def make_room(motor: str, direction: int, position: int) -> None:
            """Re-home a joint that ran out of encoder towards `direction` at `position`: its stop
            found the other way goes to the far end of the encoder, or with none found yet, where
            it is goes to the middle."""
            if (other := stops.get(-direction)) is None:
                rehome(motor, position, middles[motor])
            else:
                edge = guard + margin if direction == 1 else 2 * middles[motor] + 1 - guard - margin
                rehome(motor, other[0], edge)

        def seek(motor: str, direction: int, report: dict[str, int]) -> int:
            # A joint that runs out of encoder is re-homed to make room and pushed on: its start
            # need not be within half a turn of its stops.
            follow_state["motion"] = direction
            for _ in range(2):
                try:
                    reached = self.find_hard_stop(
                        motor,
                        direction,
                        step=step,
                        stall_error=stall_error,
                        stall_cycles=stall_cycles,
                        period_s=period_s,
                        guard=guard,
                        load_stall=load_stall,
                        report=report,
                        on_step=on_step(motor),
                    )
                    return reached - shifts[motor]
                except EncoderEndError as end:
                    make_room(motor, direction, end.position - shifts[motor])
            raise RuntimeError(f"Motor '{motor}' found no hard stop within a turn.")

        free_step = move_step or step

        def gently(motor: str, target: int, escalate: bool = True, **kwargs) -> int:
            """Move at `move_torque_limit`, and again at the motor's own limit only if blocked."""
            go = partial(
                self.move_to_position, motor, target, step=free_step, period_s=period_s,
                tolerance=return_tolerance, **kwargs,
            )  # fmt: skip
            if move_torque_limit is None or not self._has_register(motor, "Torque_Limit"):
                return go()
            own = int(self.read("Torque_Limit", motor, normalize=False))
            self.write("Torque_Limit", motor, min(own, move_torque_limit), normalize=False)
            try:
                return go()
            except RuntimeError:
                if not escalate:
                    position = int(self.read("Present_Position", motor, normalize=False))
                    self.write("Goal_Position", motor, position, normalize=False)
                    raise
                logger.info(f"{motor}: blocked at torque limit {move_torque_limit}, moving at {own}")
                self.write("Torque_Limit", motor, own, normalize=False)
                return go()
            finally:
                self.write("Torque_Limit", motor, own, normalize=False)

        def move(motor: str, position: int) -> None:
            current = int(self.read("Present_Position", motor, normalize=False)) - shifts[motor]
            follow_state["motion"] = (position > current) - (position < current)
            gently(motor, position + shifts[motor], on_step=on_step(motor))

        def move_raw(motor: str, target: int, settle: bool = False) -> int:
            return gently(motor, target, settle=settle)

        def raise_move(motor: str, target: int, settle: bool = False) -> int:
            # A raise blocked at the move limit stops there: it presses on whatever it met no harder.
            return gently(motor, target, escalate=False, settle=settle)

        def put_back(motor: str, goal: int) -> None:
            """Return a released joint to its goal no harder than the strongest push: a joint that
            meets something on the way, as a bent hand the table, stays where it was stopped."""
            limit = max(torque_limits) if torque_limits else None
            if move_torque_limit is not None:
                limit = move_torque_limit if limit is None else min(limit, move_torque_limit)
            if limit is None or not self._has_register(motor, "Torque_Limit"):
                gently(motor, goal)
                return
            own = int(self.read("Torque_Limit", motor, normalize=False))
            self.write("Torque_Limit", motor, min(own, limit), normalize=False)
            try:
                self.move_to_position(
                    motor, goal, step=free_step, period_s=period_s, tolerance=return_tolerance
                )
            except RuntimeError:
                position = int(self.read("Present_Position", motor, normalize=False))
                self.write("Goal_Position", motor, position, normalize=False)
                logger.info(f"{motor}: stopped {abs(goal - position)} ticks short of its goal, left there")
            finally:
                self.write("Torque_Limit", motor, own, normalize=False)

        def release(pushed: str) -> dict[str, int]:
            """Let go, where they are, of the joints pushed away from their goals; their goals."""
            released = {}
            for other in motor_names:
                if other == pushed or other in followers or other in towards:
                    continue
                goal = int(self.read("Goal_Position", other, normalize=False))
                present = int(self.read("Present_Position", other, normalize=False))
                if release_ticks is not None and abs(present - goal) > release_ticks:
                    self.write("Goal_Position", other, present, normalize=False)
                    released[other] = goal
            if released:
                logger.info(f"{pushed}: let go of {sorted(released)}, pushed from their goals")
            return released

        limited = (
            [m for m in motor_names if self._has_register(m, "Torque_Limit")] if working_torque_limit else []
        )
        own_limits = {m: int(self.read("Torque_Limit", m, normalize=False)) for m in limited}
        try:
            # Hold every joint where it is, so the arm does not sag while one is searched.
            starts = {
                motor: int(self.read("Present_Position", motor, normalize=False)) for motor in motor_names
            }
            for motor, position in starts.items():
                self.write("Goal_Position", motor, position, normalize=False)
            # The working limit caps every motor for the whole calibration, holding and moving the
            # arm as much as pushing it: a move that meets something presses no harder than that.
            for motor in limited:
                self.write("Torque_Limit", motor, working_torque_limit, normalize=False)
            self.enable_torque(motor_names)

            # Raise: before anything else moves, the arm is lifted clear of the table it may rest
            # on, one joint after the other, each surveyed first: a hand resting on the table
            # carries the arm's weight, which hides the pull of gravity on the joints above it.
            gravity: dict[str, int] = {}
            raised_from: dict[str, int] = {}
            for motor, mode in (raise_arm or {}).items():
                if motor not in searched:
                    continue
                gravity[motor] = self._survey_gravity(motor, survey_ticks, gravity_threshold, move_raw)
                if gravity[motor]:
                    raised_from[motor] = starts[motor]
                    if not self._raise_joint(
                        motor, gravity[motor], mode, raise_move, threshold=gravity_threshold
                    ):
                        # Blocked at once, the joint was turning down onto what it rests on, as a hand
                        # on the table, which also misled the survey: gravity pulls the other way.
                        gravity[motor] = -gravity[motor]
                        logger.info(f"{motor}: blocked at once, gravity pulls the other way")
                        self._raise_joint(
                            motor, gravity[motor], mode, raise_move, threshold=gravity_threshold
                        )
                    starts[motor] = int(self.read("Present_Position", motor, normalize=False)) - shifts[motor]

            # Survey: turn each joint a little each way to learn where gravity pulls it.
            for motor in searched:
                if motor not in gravity:
                    gravity[motor] = self._survey_gravity(motor, survey_ticks, gravity_threshold, move_raw)

            # Lift: links turned up a little, away from the table, for the searches before their own.
            lifted_from: dict[str, int] = {}
            for motor, ticks in (lift or {}).items():
                if gravity.get(motor):
                    lifted_from[motor] = starts[motor]
                    try:
                        move(motor, starts[motor] - gravity[motor] * ticks)
                    except RuntimeError:
                        logger.info(f"{motor}: could not lift {ticks} ticks")
                    starts[motor] = int(self.read("Present_Position", motor, normalize=False)) - shifts[motor]

            # Each joint in turn, then the ones revisited, in postures of joints calibrated since.
            schedule = [(motor, postures.get(motor) or [{}]) for motor in searched] + list(revisit.items())
            found: dict[str, dict[int, tuple[int, bool]]] = {}

            def full_range(motor: str, ends: Mapping[int, tuple[int, bool]]) -> bool:
                """Whether the stops found are as far apart as the range expected of the joint: both
                are its stops, as a stall short of either would leave the range short, and no other
                posture, nor a revisit, could find it wider."""
                expected = (expected_ranges or {}).get(motor)
                if expected is None or -1 not in ends or 1 not in ends:
                    return False
                span = ends[1][0] - ends[-1][0]
                if span < (1 - range_tolerance) * expected:
                    return False
                logger.info(f"{motor}: range of {span} ticks, as wide as expected ({expected})")
                return True

            for motor, motor_postures in schedule:
                if motor in found and full_range(motor, found[motor]):
                    continue
                if motor in raised_from:
                    # Its own search starts where the arm was put: the raise only kept the hand
                    # off the table for the joints searched before.
                    starts[motor] = raised_from.pop(motor)
                    move(motor, starts[motor])
                stops.clear()
                stops.update(found.get(motor, {}))
                # A joint to raise whose survey found no pull, as one standing near upright when the
                # arm started, is surveyed again in the posture, from the middle of its range.
                for posture in motor_postures:
                    unknown = [
                        o for o, place in posture.items() if isinstance(place, Raised) and not gravity.get(o)
                    ]
                    if not unknown:
                        continue
                    placed = {
                        o: place
                        for o, place in posture.items()
                        if not isinstance(place, (Follow, Raised, Toward, Turn))
                    }
                    for other, fraction in placed.items():
                        low, high = range_mins[other], range_maxes[other]
                        move(other, round(low + fraction * (high - low)) - shifts[other])
                    for other in unknown:
                        move(other, (range_mins[other] + range_maxes[other]) // 2 - shifts[other])
                        gravity[other] = self._survey_gravity(
                            other, survey_ticks, gravity_threshold, move_raw
                        )
                        move(other, starts[other])
                    for other in placed:
                        move(other, starts[other])
                for posture in self._resolve_raised(motor_postures, gravity, motor):
                    for other, place in posture.items():
                        if not isinstance(place, (Follow, Raised, Toward, Turn)):
                            low, high = range_mins[other], range_maxes[other]
                            move(other, round(low + place * (high - low)) - shifts[other])
                    follow_state["turned"] = 0
                    for other, place in posture.items():
                        if isinstance(place, Follow) and place.lift and gravity.get(other):
                            move(other, starts[other] - gravity[other] * place.lift)
                    followers.update(
                        {
                            other: (
                                place.ratio,
                                int(self.read("Present_Position", other, normalize=False)) - shifts[other],
                            )
                            for other, place in posture.items()
                            if isinstance(place, Follow)
                        }
                    )
                    for other, place in posture.items():
                        if isinstance(place, Toward) and gravity.get(other) and gravity.get(motor):
                            low, high = range_mins[other], range_maxes[other]

                            def at(
                                fraction: float, low: int = low, high: int = high, other: str = other
                            ) -> int:
                                # From the end gravity pulls the motor towards.
                                if gravity[other] < 0:
                                    return round(low + fraction * (high - low))
                                return round(high - fraction * (high - low))

                            with_gravity, against = at(place.down), at(place.up)
                            towards[other] = (
                                int(self.read("Present_Position", other, normalize=False)) - shifts[other],
                                with_gravity if gravity[motor] < 0 else against,
                                against if gravity[motor] < 0 else with_gravity,
                                place.reach,
                            )
                        if isinstance(place, Turn) and gravity.get(other) and gravity.get(motor):
                            here = int(self.read("Present_Position", other, normalize=False))
                            # Turned from where the arm was put, not from where a raise left it.
                            origin = raised_from.get(other, here - shifts[other]) + shifts[other]
                            with_gravity = origin - gravity[other] * place.down
                            against = origin - gravity[other] * place.up
                            towards[other] = (
                                here - shifts[other],
                                with_gravity if gravity[motor] < 0 else against,
                                against if gravity[motor] < 0 else with_gravity,
                                place.reach,
                            )
                    for direction in (-1, 1):
                        known = stops.get(direction)
                        stop, confirmed = self._seek_confirmed_stop(
                            motor,
                            direction,
                            torque_limits,
                            confirm_tolerance,
                            seek,
                            trust_impacts=trust_impacts,
                            known=known[0] if known is not None and known[1] else None,
                        )
                        best = stops.get(direction)
                        if best is None or direction * (stop - best[0]) > 0:
                            stops[direction] = (stop, confirmed)
                        if direction == -1 and 1 not in stops:
                            # The low stop near the bottom of the encoder leaves the high stop room.
                            rehome(motor, stops[-1][0], guard + margin)
                        released = release(motor) if release_ticks is not None else {}
                        move(motor, starts[motor])
                        for other, goal in released.items():
                            put_back(other, goal)
                    # The range found so far centred on the encoder leaves room both ways for the
                    # next posture, should it reach further, as a joint once too weak to lift.
                    rehome(motor, (stops[-1][0] + stops[1][0]) // 2, middles[motor])
                    followers.clear()
                    towards.clear()
                    for other in posture:
                        move(other, starts[other])
                    logger.info(f"{motor}: stops at {stops[-1][0]} and {stops[1][0]} in posture {posture}")
                    if full_range(motor, stops):
                        break

                found[motor] = dict(stops)
                if motor in lifted_from:
                    # Searched now: back down to where it started, for the searches after its own.
                    starts[motor] = lifted_from.pop(motor)
                    move(motor, starts[motor])
                (low, _), (high, _) = stops[-1], stops[1]
                if high - low <= 2 * margin:
                    raise RuntimeError(
                        f"Motor '{motor}' moved only {high - low} ticks between its stops: is it blocked?"
                    )
                # Like a manual calibration, the middle of the range is a half turn.
                rehome(motor, (low + high) // 2, middles[motor])
                homing_offsets[motor] = int(self.read("Homing_Offset", motor, normalize=False))
                range_mins[motor] = low + shifts[motor] + margin
                range_maxes[motor] = high + shifts[motor] - margin
                logger.info(
                    f"{motor}: hard stops at {range_mins[motor] - margin} and {range_maxes[motor] + margin}"
                )
        finally:
            self.disable_torque(motor_names)
            for motor, own_limit in own_limits.items():
                self.write("Torque_Limit", motor, own_limit, normalize=False)

        unconfirmed = [
            f"{motor} {'low' if direction < 0 else 'high'}"
            for motor, motor_stops in found.items()
            for direction, (_, confirmed) in sorted(motor_stops.items())
            if not confirmed
        ]
        if unconfirmed:
            logger.warning(
                f"These stops were still moving at the strongest push, so the ranges may be short: {unconfirmed}."
                " Raise the strongest torque limit, or check what holds these joints back."
            )
        for motor, expected in (expected_ranges or {}).items():
            if motor in range_mins:
                span = range_maxes[motor] - range_mins[motor] + 2 * margin
                if abs(span - expected) > 0.05 * expected:
                    logger.warning(
                        f"{motor}: range of {span} ticks, {'short of' if span < expected else 'beyond'} the "
                        f"{expected} expected. Check what held it back, or correct the expected range."
                    )

        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = self.resolution(self.motors[motor].model) - 1

        return {
            motor: MotorCalibration(
                id=self.motors[motor].id,
                drive_mode=drive_modes[motor],
                homing_offset=int(homing_offsets[motor]),
                range_min=int(range_mins[motor]),
                range_max=int(range_maxes[motor]),
            )
            for motor in self.motors
            if motor in motor_names
        }

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self._inverts_in_software(motor) and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[id_] = -norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[id_] = 100 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.resolution(self._id_to_model(id_)) - 1
                normalized_values[id_] = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError

        return normalized_values

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self._inverts_in_software(motor) and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.resolution(self._id_to_model(id_)) - 1
                unnormalized_values[id_] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError

        return unnormalized_values

    @check_if_not_connected
    def ping(self, motor: NameOrID, num_retry: int = 0) -> int | None:
        """The model number a motor answers with.

        Reads `Model_Number` rather than sending a ping: presence and identity then cost one
        round trip instead of two.

        Args:
            motor (`str | int`):
                Motor, by name or id, one of the bus' motors.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a communication failure.

        Returns:
            `int | None`: The model number, `None` when the motor does not answer or refuses the read.
        """
        try:
            return self._read("Model_Number", self._get_motor_id(motor), num_retry)
        except (ConnectionError, RuntimeError):
            return None

    @check_if_not_connected
    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Value:
        """Read a register of one motor.

        Args:
            data_name (`str`):
                Register name, e.g. `"Present_Position"`; the case does not matter.
            motor (`str`):
                Motor name.
            normalize (`bool`, *optional*, defaults to `True`):
                Whether to scale a position to the range its calibration and norm mode define.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a communication failure.

        Returns:
            `int | float`: The raw value, or the normalised one.

        Raises:
            ConnectionError: If the motor does not answer.
            RuntimeError: If the motor refuses the read. An alarm in its status, as an
                overload, does not raise.
        """

        id_ = self.motors[motor].id
        value = self._read(data_name, id_, num_retry)

        if normalize and data_name.lower() in self.normalized_data:
            return self._normalize({id_: value})[id_]

        return value

    def _read(self, data_name: str, motor_id: int, num_retry: int) -> int:
        """Read one register of a motor by id: `read` takes a name, `ping` any id."""
        try:
            value, status = self._connected_bus.read_register_with_error(
                motor_id, data_name.lower(), retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(
                f"Failed to read '{data_name}' from motor ID {motor_id} after {num_retry + 1} tries. {e}"
            ) from e

        if self._definitions[motor_id].refuses(status):
            raise RuntimeError(
                f"Failed to read '{data_name}': motor ID {motor_id} refused it with error status 0x{status:02x}."
            )

        return value

    @check_if_not_connected
    def write(
        self, data_name: str, motor: str, value: Value, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        """Write a register of one motor, and wait for its answer.

        Unlike [`~SerialMotorsBus.sync_write`], the motor acknowledges the write, so a failure is
        reported: slower, but suited to configuring motors.

        Args:
            data_name (`str`):
                Register name, e.g. `"Goal_Position"`; the case does not matter.
            motor (`str`):
                Motor name.
            value (`int | float`):
                Value to write, normalised or raw as `normalize` says.
            normalize (`bool`, *optional*, defaults to `True`):
                Whether a position is given in the range its calibration and norm mode define.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a communication failure.

        Raises:
            ConnectionError: If the motor does not answer.
            RuntimeError: If the motor refuses the write. An alarm in its status, as an
                overload, does not raise.
        """

        id_ = self.motors[motor].id
        int_value = int(value)
        if normalize and data_name.lower() in self.normalized_data:
            int_value = self._unnormalize({id_: value})[id_]

        try:
            status = self._connected_bus.write_register_with_error(
                id_, data_name.lower(), int_value, retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(
                f"Failed to write {int_value} to '{data_name}' of motor ID {id_} after {num_retry + 1} tries. {e}"
            ) from e

        if self._definitions[id_].refuses(status):
            raise RuntimeError(
                f"Failed to write {int_value} to '{data_name}': motor ID {id_} refused it with error status "
                f"0x{status:02x}."
            )

    @check_if_not_connected
    def sync_read(
        self,
        data_name: str,
        motors: NameOrID | Sequence[NameOrID] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        """Read a register of several motors.

        rustypot sends one Sync Read per group of motors sharing a protocol and the register's
        address and size, and reads the motors of a servo without Sync Read, as the Feetech SCS,
        one after the other. The values are therefore not one simultaneous snapshot.

        Args:
            data_name (`str`):
                Register name, e.g. `"Present_Position"`; the case does not matter.
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to read, by name or id. `None` reads every motor.
            normalize (`bool`, *optional*, defaults to `True`):
                Whether to scale positions to the range their calibration and norm mode define.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a communication failure. A retry sends every group
                again, those that had succeeded included; an unknown motor or register is never
                retried.

        Returns:
            `dict[str, int | float]`: The value of each motor, by name.

        Raises:
            ConnectionError: If a motor does not answer.
            RuntimeError: If a motor refuses the read. An alarm in a status, as an overload,
                does not raise.
        """

        names = self._get_motors_list(motors)
        ids = [self._get_motor_id(motor) for motor in names]

        try:
            read = self._connected_bus.sync_read_register_with_error(
                ids, data_name.lower(), retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(
                f"Failed to sync read '{data_name}' from motor IDs {ids} after {num_retry + 1} tries. {e}"
            ) from e
        if errors := [
            f"motor ID {id_} refused it with error status 0x{status:02x}"
            for id_, (_, status) in zip(ids, read, strict=True)
            if self._definitions[id_].refuses(status)
        ]:
            raise RuntimeError(f"Failed to sync read '{data_name}': {', '.join(errors)}.")
        ids_values = {id_: value for id_, (value, _) in zip(ids, read, strict=True)}

        if normalize and data_name.lower() in self.normalized_data:
            return {self._id_to_name(id_): value for id_, value in self._normalize(ids_values).items()}

        return {self._id_to_name(id_): value for id_, value in ids_values.items()}

    @check_if_not_connected
    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """Write a register of several motors, without waiting for an answer.

        Unlike [`~SerialMotorsBus.write`], no motor answers a sync write: faster, suited to
        control loops, but a lost packet goes unnoticed. rustypot sends one Sync Write per group
        of motors sharing a protocol and the register's address and size.

        Args:
            data_name (`str`):
                Register name, e.g. `"Goal_Position"`; the case does not matter.
            values (`int | float | dict[str, int | float]`):
                One value for every motor, or a value per motor name.
            normalize (`bool`, *optional*, defaults to `True`):
                Whether positions are given in the range their calibration and norm mode define.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a failure of the port. A retry sends every group again.

        Raises:
            ConnectionError: If the port fails.
        """

        raw_ids_values = self._get_ids_values_dict(values)
        int_ids_values = {id_: int(val) for id_, val in raw_ids_values.items()}
        if normalize and data_name.lower() in self.normalized_data:
            int_ids_values = self._unnormalize(raw_ids_values)

        try:
            self._connected_bus.sync_write_register(
                list(int_ids_values), data_name.lower(), list(int_ids_values.values()), retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(
                f"Failed to sync write '{data_name}' to motor IDs {list(int_ids_values)} after "
                f"{num_retry + 1} tries. {e}"
            ) from e
