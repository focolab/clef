"""Tests for configure-time intensity on light-source output devices."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))

# Mock pycromanager before importing the modules
sys.modules["pycromanager"] = MagicMock()

from apps.io.output_device.ldi_89north_output import LDI89NorthOutput
from apps.io.output_device.mm_dac_lightsource import MMDACLightSourceOutput


def _make_ldi(config):
    dev = LDI89NorthOutput(name="ldi", config=config)
    dev.mmc = MagicMock()
    return dev


def _make_dac(config):
    dev = MMDACLightSourceOutput(name="dac", config=config)
    dev.mmc = MagicMock()
    return dev


class TestLDIConfigureIntensity:
    """LDI89NorthOutput.configure applies a configured intensity."""

    def test_applies_configured_intensity(self):
        dev = _make_ldi({
            "intensity_device": "LDI",
            "intensity_property": "470 Intensity",
            "intensity": 50,
        })
        dev.configure()
        dev.mmc.setProperty.assert_called_once_with("LDI", "470 Intensity", 50)

    def test_defaults_to_zero_when_absent(self):
        dev = _make_ldi({
            "intensity_device": "LDI",
            "intensity_property": "470 Intensity",
        })
        dev.configure()
        dev.mmc.setProperty.assert_called_once_with("LDI", "470 Intensity", 0)


class TestDACConfigureIntensity:
    """MMDACLightSourceOutput.configure applies a configured intensity, clamped."""

    def test_applies_configured_intensity(self):
        dev = _make_dac({
            "dac_device": "DAC488",
            "dac_property": "Volts",
            "max_volts": 3.5,
            "intensity": 1.5,
        })
        dev.configure()
        dev.mmc.setProperty.assert_called_once_with("DAC488", "Volts", 1.5)

    def test_defaults_to_zero_when_absent(self):
        dev = _make_dac({
            "dac_device": "DAC488",
            "dac_property": "Volts",
        })
        dev.configure()
        dev.mmc.setProperty.assert_called_once_with("DAC488", "Volts", 0.0)

    def test_clamps_to_max_volts(self):
        dev = _make_dac({
            "dac_device": "DAC488",
            "dac_property": "Volts",
            "max_volts": 3.5,
            "intensity": 99.0,
        })
        dev.configure()
        dev.mmc.setProperty.assert_called_once_with("DAC488", "Volts", 3.5)
