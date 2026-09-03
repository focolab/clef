"""Tests for tying fluorescence excitation to the recording button.

The target moves faster under excitation, so it is found and centered under
brightfield with the light at 0, and the light only comes up when recording
starts. Getting this wrong either bleaches the sample for the whole session or
leaves it dark during the recording, so the on/off edges are worth pinning down.
"""

import pytest

from apps.io.output_device.mm_property_lightsource import (
    MMPropertyLightSourceOutput,
)
from apps.logic.xy_tracking_logic import XYTrackingLogic


class FakeCore:
    """Minimal Micro-Manager stand-in that records property writes."""

    def __init__(self, devices=None):
        # device label -> set of property names it exposes
        self.devices = devices if devices is not None else {
            "Camera": {"Binning"},
            "LightEngine": {"LightEngineIntensity", "State"},
            "XYStage": {"StepSize"},
        }
        self.writes = []

    def getLoadedDevices(self):
        return list(self.devices)

    def hasProperty(self, device, prop):
        return prop in self.devices[device]

    def setProperty(self, device, prop, value):
        self.writes.append((device, prop, value))

    def setShutterOpen(self, device, state):
        self.writes.append((device, "shutter", state))


def make_light(config=None, core=None):
    dev = MMPropertyLightSourceOutput("fluorescence_light", config or {
        "intensity_device": None,
        "intensity_property": "LightEngineIntensity",
    })
    dev.mmc = core or FakeCore()
    dev.configure()
    return dev


class TestDeviceDiscovery:
    """Device labels differ between rigs; the property name does not."""

    def test_finds_the_device_carrying_the_property(self):
        dev = make_light()
        assert dev._intensity_device == "LightEngine"

    def test_starts_at_zero(self):
        """A session must never begin by illuminating the sample."""
        core = FakeCore()
        make_light(core=core)
        assert core.writes == [("LightEngine", "LightEngineIntensity", 0)]

    def test_explicit_label_is_not_overridden(self):
        core = FakeCore()
        dev = make_light({"intensity_device": "LightEngine",
                          "intensity_property": "LightEngineIntensity"}, core)
        assert dev._intensity_device == "LightEngine"

    def test_missing_property_disables_control_without_raising(self):
        """A rig without programmatic light control must still run."""
        core = FakeCore({"Camera": {"Binning"}})
        dev = make_light(core=core)
        assert dev._intensity_device is None
        dev.update_output(intensity=100)
        assert core.writes == []

    def test_sets_intensity(self):
        core = FakeCore()
        dev = make_light(core=core)
        core.writes.clear()
        dev.update_output(intensity=100)
        assert core.writes == [("LightEngine", "LightEngineIntensity", 100.0)]

    def test_intensity_is_clamped(self):
        """This drives illumination onto a live sample; out-of-range is a
        photodamage risk, not a rounding detail."""
        core = FakeCore()
        dev = make_light({"intensity_device": None,
                          "intensity_property": "LightEngineIntensity",
                          "max_intensity": 100}, core)
        core.writes.clear()
        dev.update_output(intensity=5000)
        dev.update_output(intensity=-10)
        assert core.writes == [
            ("LightEngine", "LightEngineIntensity", 100.0),
            ("LightEngine", "LightEngineIntensity", 0.0),
        ]

    def test_close_leaves_the_source_dark(self):
        core = FakeCore()
        dev = make_light(core=core)
        dev.update_output(intensity=100)
        core.writes.clear()
        dev.close()
        assert core.writes == [("LightEngine", "LightEngineIntensity", 0)]


class FakeLight:
    """Stands in for the light output device as the logic sees it."""

    def __init__(self):
        self.levels = []

    def update_output(self, **kwargs):
        self.levels.append(kwargs.get("intensity"))


@pytest.fixture
def logic():
    """Logic with a fake light attached and no worker subprocess."""
    light = FakeLight()
    lg = XYTrackingLogic(
        "xy_tracking",
        {"light_output_name": "fluorescence_light",
         "acquisition_intensity": 100, "idle_intensity": 0},
        output_devices={"fluorescence_light": light},
    )
    return lg, light


def send(logic, *messages):
    """Feed worker messages through the real event drain."""
    class Conn:
        def __init__(self, msgs):
            self.msgs = list(msgs)

        def poll(self):
            return bool(self.msgs)

        def recv(self):
            return self.msgs.pop(0)

    logic.parent_conn = Conn(messages)
    logic._drain_events()
    logic.parent_conn = None


class TestExcitationFollowsRecording:

    def test_light_comes_up_on_record_start(self, logic):
        lg, light = logic
        send(lg, {"type": "recording_start", "index": 0, "name": "sub00",
                  "start_frame": 10, "start_ts": 1.0, "intensity": 100})
        assert light.levels == [100]

    def test_light_goes_off_on_record_stop(self, logic):
        lg, light = logic
        send(lg,
             {"type": "recording_start", "index": 0, "name": "sub00",
              "start_frame": 10, "start_ts": 1.0, "intensity": 100},
             {"type": "recording_stop", "index": 0, "name": "sub00",
              "start_frame": 10, "stop_frame": 99, "start_ts": 1.0,
              "stop_ts": 2.0, "intensity": 100})
        assert light.levels == [100, 0]

    def test_worker_chosen_level_is_honoured(self, logic):
        """The GUI spinbox, not the config default, decides the level."""
        lg, light = logic
        send(lg, {"type": "recording_start", "index": 0, "name": "sub00",
                  "start_frame": 0, "start_ts": 1.0, "intensity": 45})
        assert light.levels == [45]

    def test_level_is_recorded_in_the_epoch_metadata(self, logic):
        lg, light = logic
        send(lg,
             {"type": "recording_start", "index": 0, "name": "sub00",
              "start_frame": 10, "start_ts": 1.0, "intensity": 60},
             {"type": "recording_stop", "index": 0, "name": "sub00",
              "start_frame": 10, "stop_frame": 99, "start_ts": 1.0,
              "stop_ts": 2.0, "intensity": 60})
        assert lg.get_metadata()["sub_acquisitions"][0]["intensity"] == 60

    def test_stays_dark_until_recording_starts(self, logic):
        """Nothing else may raise the excitation - that is the whole point."""
        lg, light = logic
        send(lg, {"type": "calibration_update", "micron_to_pix_ratio": 0.9})
        assert light.levels == []

    def test_close_returns_the_light_to_idle(self, logic):
        """A session that ends mid-recording must not leave the sample lit."""
        lg, light = logic
        send(lg, {"type": "recording_start", "index": 0, "name": "sub00",
                  "start_frame": 10, "start_ts": 1.0, "intensity": 100})
        lg.close()
        assert light.levels[-1] == 0

    def test_missing_light_device_is_survivable(self):
        """Tracking is useful on a rig with no programmatic light control."""
        lg = XYTrackingLogic("xy_tracking", {}, output_devices={})
        send(lg, {"type": "recording_start", "index": 0, "name": "sub00",
                  "start_frame": 0, "start_ts": 1.0, "intensity": 100})
        lg.close()  # must not raise

    def test_a_failing_light_does_not_abort_the_recording(self):
        """Losing the light mid-session should not take the session with it."""
        class Broken:
            def update_output(self, **kwargs):
                raise RuntimeError("bridge died")

        lg = XYTrackingLogic(
            "xy_tracking", {"light_output_name": "fluorescence_light"},
            output_devices={"fluorescence_light": Broken()},
        )
        send(lg, {"type": "recording_start", "index": 0, "name": "sub00",
                  "start_frame": 0, "start_ts": 1.0, "intensity": 100})
        assert lg.open_epoch["name"] == "sub00"
