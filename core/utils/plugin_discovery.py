"""
Plugin discovery helper for CLEF.

Imports the ``.py`` files in a plugin directory so that any device, data
interface, or logic subclass they define is registered via ``__init_subclass__``.

Modules are loaded directly from their file paths (``spec_from_file_location``)
rather than by computing a dotted ``apps.*`` import name. This decouples plugin
discovery from the package name and the directory's location on ``sys.path``,
so plugin directories may live outside the CLEF tree and be named anything.
"""

import hashlib
import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _unique_module_name(py_file: Path) -> str:
    """Build a collision-resistant module name from the file's absolute path.

    Using the absolute path (hashed) keeps modules with the same filename in
    different plugin directories from clobbering each other in ``sys.modules``.
    """
    resolved = py_file.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"clef_plugin_{resolved.stem}_{digest}"


def import_plugins_from(directory: Path) -> None:
    """Import all non-underscore ``.py`` files in ``directory``.

    Failures importing an individual plugin are logged and skipped so one bad
    module does not prevent the rest from loading.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.debug(f"Plugin directory does not exist: {directory}")
        return

    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            mod_name = _unique_module_name(py_file)
            spec = importlib.util.spec_from_file_location(mod_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not create import spec for plugin {py_file}")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.debug(f"Loaded plugin module: {py_file}")
        except Exception as e:
            logger.warning(f"Failed to import plugin {py_file}: {e}")
