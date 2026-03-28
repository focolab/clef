"""Tests for EventLog."""

import time
import pytest
from core.utils.event_log import EventLog


class TestEventLog:

    def test_record_appends_event(self):
        log = EventLog()
        log.record("a")
        log.record("b")
        assert len(log.events) == 2
        assert log.events[0] == (0, log.events[0][1], "a")
        assert log.events[1] == (1, log.events[1][1], "b")

    def test_timestamps_increase(self):
        log = EventLog()
        log.record()
        log.record()
        assert log.timestamps[1] >= log.timestamps[0]

    def test_record_returns_timestamp(self):
        log = EventLog()
        t = log.record()
        assert isinstance(t, float)
        assert t >= 0

    def test_disabled_does_not_record(self):
        log = EventLog()
        log.enabled = False
        t = log.record("x")
        assert t == -1.0
        assert len(log.events) == 0

    def test_to_dict_structure(self):
        log = EventLog()
        log.record("payload")
        d = log.to_dict()
        assert "t0_wall" in d
        assert "events" in d
        assert isinstance(d["t0_wall"], float)
        assert len(d["events"]) == 1

    def test_timestamps_property(self):
        log = EventLog()
        log.record("a")
        log.record("b")
        ts = log.timestamps
        assert len(ts) == 2
        assert all(isinstance(t, float) for t in ts)

    def test_empty_log(self):
        log = EventLog()
        assert log.events == []
        assert log.timestamps == []
        d = log.to_dict()
        assert d["events"] == []

    def test_none_payload_default(self):
        log = EventLog()
        log.record()
        assert log.events[0][2] is None

    def test_sample_indices(self):
        log = EventLog()
        log.record("a")
        log.record("b")
        log.record("c")
        assert log.sample_indices == [0, 1, 2]


class TestEventLogOnBaseInputDevice:

    def test_get_input_auto_records(self):
        from core.io.input_device.BaseInputDevice import BaseInputDevice
        dev = BaseInputDevice(name="test")
        dev.get_input()
        dev.get_input()
        assert len(dev.event_log.events) == 2

    def test_event_log_in_metadata(self):
        from core.io.input_device.BaseInputDevice import BaseInputDevice
        dev = BaseInputDevice(name="test")
        dev.get_input()
        meta = dev.get_metadata()
        assert "event_log" in meta
        assert len(meta["event_log"]["events"]) == 1

    def test_disabled_event_log(self):
        from core.io.input_device.BaseInputDevice import BaseInputDevice
        dev = BaseInputDevice(name="test")
        dev.event_log.enabled = False
        dev.get_input()
        assert len(dev.event_log.events) == 0


class TestEventLogOnBaseOutputDevice:

    def test_update_output_auto_records(self):
        from core.io.output_device.BaseOutputDevice import BaseOutputDevice
        dev = BaseOutputDevice(name="test")
        dev.update_output(x=1)
        dev.update_output(y=2)
        assert len(dev.event_log.events) == 2
        assert dev.event_log.events[0][2] == {"x": 1}
        assert dev.event_log.events[1][2] == {"y": 2}

    def test_event_log_in_metadata(self):
        from core.io.output_device.BaseOutputDevice import BaseOutputDevice
        dev = BaseOutputDevice(name="test")
        dev.update_output(a=1)
        meta = dev.get_metadata()
        assert "event_log" in meta
        assert len(meta["event_log"]["events"]) == 1


class TestEventLogOnBaseClosedLoopLogic:

    def test_check_logic_does_not_record_none(self):
        from core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic
        logic = BaseClosedLoopLogic(name="test")
        logic.check_logic()
        assert len(logic.event_log.events) == 0

    def test_event_log_in_metadata(self):
        from core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic
        logic = BaseClosedLoopLogic(name="test")
        meta = logic.get_metadata()
        assert "event_log" in meta
