"""
Tests for core.utils.plugin_discovery.

Verifies that plugin modules are discovered and imported from an arbitrary
external directory — one not named ``apps`` and not on ``sys.path`` — so that
private plugin repositories can live outside the CLEF tree.
"""

import textwrap
from pathlib import Path

from core.io.input_device.BaseInputDevice import BaseInputDevice
from core.utils.plugin_discovery import import_plugins_from


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(body))
    return path


def test_imports_module_from_external_named_dir(tmp_path):
    """A plugin in a dir not named 'apps' registers its subclass."""
    plugin_dir = tmp_path / "my_private_apps" / "io"
    plugin_dir.mkdir(parents=True)
    _write(
        plugin_dir,
        "ext_device.py",
        """
        from core.io.input_device.BaseInputDevice import BaseInputDevice

        class ExtInputDevice(BaseInputDevice):
            device_class = "ext_input_device_test"
        """,
    )

    import_plugins_from(plugin_dir)

    assert "ext_input_device_test" in BaseInputDevice._registry


def test_skips_underscore_files(tmp_path):
    """Files starting with underscore are not imported."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write(
        plugin_dir,
        "_private.py",
        """
        from core.io.input_device.BaseInputDevice import BaseInputDevice

        class ShouldNotRegister(BaseInputDevice):
            device_class = "should_not_register_test"
        """,
    )

    import_plugins_from(plugin_dir)

    assert "should_not_register_test" not in BaseInputDevice._registry


def test_same_filename_different_dirs_both_load(tmp_path):
    """Two plugins with the same filename in different dirs both register."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write(
        dir_a,
        "device.py",
        """
        from core.io.input_device.BaseInputDevice import BaseInputDevice

        class DeviceA(BaseInputDevice):
            device_class = "collision_device_a_test"
        """,
    )
    _write(
        dir_b,
        "device.py",
        """
        from core.io.input_device.BaseInputDevice import BaseInputDevice

        class DeviceB(BaseInputDevice):
            device_class = "collision_device_b_test"
        """,
    )

    import_plugins_from(dir_a)
    import_plugins_from(dir_b)

    assert "collision_device_a_test" in BaseInputDevice._registry
    assert "collision_device_b_test" in BaseInputDevice._registry


def test_missing_directory_is_noop(tmp_path):
    """Pointing at a nonexistent directory does not raise."""
    import_plugins_from(tmp_path / "does_not_exist")


def test_bad_plugin_does_not_block_others(tmp_path):
    """A module that raises on import is skipped; others still load."""
    plugin_dir = tmp_path / "mixed"
    plugin_dir.mkdir()
    _write(plugin_dir, "broken.py", "raise RuntimeError('boom')\n")
    _write(
        plugin_dir,
        "good.py",
        """
        from core.io.input_device.BaseInputDevice import BaseInputDevice

        class GoodDevice(BaseInputDevice):
            device_class = "good_after_bad_test"
        """,
    )

    import_plugins_from(plugin_dir)

    assert "good_after_bad_test" in BaseInputDevice._registry
