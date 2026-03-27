"""
Shared test subclasses and fixtures for clef2 integration tests.
"""

from typing import Any, ClassVar, Dict, Optional
from clef2.core.io.input_device.BaseInputDevice import BaseInputDevice
from clef2.core.io.output_device.BaseOutputDevice import BaseOutputDevice
from clef2.core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic


# ---------------------------------------------------------------------------
# Test device / logic subclasses
# ---------------------------------------------------------------------------

class CountingInputDevice(BaseInputDevice):
    device_class: ClassVar[Optional[str]] = "counting_input"
    device_type: ClassVar[Optional[str]] = "test"

    def __init__(self, name, config=None, io_manager=None):
        super().__init__(name, config, io_manager=io_manager)
        self.counter = 0

    def get_input(self):
        self.counter += 1
        return self.counter


class TrackingOutputDevice(BaseOutputDevice):
    device_class: ClassVar[Optional[str]] = "tracking_output"
    device_type: ClassVar[Optional[str]] = "test"

    def __init__(self, name, config=None, io_manager=None):
        super().__init__(name, config, io_manager=io_manager)
        self.calls = []

    def update_output(self, **kwargs):
        self.calls.append(kwargs)


class SaveTrackingInputDevice(BaseInputDevice):
    device_class: ClassVar[Optional[str]] = "save_tracking_input"
    device_type: ClassVar[Optional[str]] = "test"

    def __init__(self, name, config=None, io_manager=None):
        super().__init__(name, config, io_manager=io_manager)
        self.save_calls = []

    def get_input(self):
        return 0

    def save_data(self, **kwargs):
        self.save_calls.append(kwargs)


class PassthroughLogic(BaseClosedLoopLogic):
    logic_class: ClassVar[Optional[str]] = "passthrough_logic"

    def __init__(self, name, config=None, output_devices=None, gui_parameters=None, **kwargs):
        super().__init__(name, config, output_devices, gui_parameters, **kwargs)
        self.process_sample_count = 0
        self.last_sample = None

    def process_sample(self, sample):
        self.process_sample_count += 1
        self.last_sample = sample

    def check_logic(self):
        return None


class CyclingLogic(BaseClosedLoopLogic):
    """Every 10 samples, targets output devices in rotation:
    cycle 0 → [output_0], cycle 1 → [output_1], cycle 2 → [output_2],
    cycle 3 → [output_0, output_1, output_2], then repeat.
    """
    logic_class: ClassVar[Optional[str]] = "cycling_logic"

    def __init__(self, name, config=None, output_devices=None, gui_parameters=None, **kwargs):
        super().__init__(name, config, output_devices, gui_parameters, **kwargs)
        self.process_sample_count = 0
        self.last_sample = None
        self._device_names = sorted(self.output_devices.keys())

    def process_sample(self, sample):
        self.process_sample_count += 1
        self.last_sample = sample

    def check_logic(self):
        count = self.process_sample_count
        if count == 0 or count % 10 != 0:
            return None

        cycle = ((count // 10) - 1) % 4
        if cycle < len(self._device_names):
            targets = [self._device_names[cycle]]
        else:
            targets = list(self._device_names)

        return {name: {"iteration": count} for name in targets}


class SaveTrackingLogic(BaseClosedLoopLogic):
    logic_class: ClassVar[Optional[str]] = "save_tracking_logic"

    def __init__(self, name, config=None, output_devices=None, gui_parameters=None, **kwargs):
        super().__init__(name, config, output_devices, gui_parameters, **kwargs)
        self.save_calls = []

    def save_data(self, **kwargs):
        self.save_calls.append(kwargs)
