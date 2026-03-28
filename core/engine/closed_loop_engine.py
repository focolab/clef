"""
ClosedLoopEngine for CLEF2.

Orchestrates the real-time closed loop:
  update_input -> process_sample -> check_logic -> update_output
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.config.config_manager import ConfigManager
from core.io.io_manager import IOManager
from core.logic.logic_manager import LogicManager
from core.utils.metadata_io import save_metadata as _save_metadata_to_file

logger = logging.getLogger(__name__)


class ClosedLoopEngine:
    """Orchestrates IO and logic managers in a closed-loop acquisition cycle."""

    def __init__(
        self,
        config_manager: ConfigManager,
        io_manager: IOManager,
        logic_manager: LogicManager,
    ):
        self.config_manager = config_manager
        self.io_manager = io_manager
        self.logic_manager = logic_manager
        self.running = False

    def loop(self, iterations: int = 1):
        """Run the closed loop for a given number of iterations.

        Pass iterations=-1 to run indefinitely until stopped externally
        (e.g. Ctrl+C or setting self.running = False).

        Each iteration:
        1. update_input — data interfaces poll devices and store samples
        2. process_sample — distribute input_stores to all logic algorithms
        3. check_logic — collect logic_update dicts from algorithms
        4. update_output — route logic updates to output devices
        """
        self.running = True
        count = 0
        while self.running:
            if iterations >= 0 and count >= iterations:
                break
            count += 1

            self.io_manager.update_input()
            self.logic_manager.process_sample(self.io_manager.input_stores)

            logic_update = self.logic_manager.check_logic()
            if logic_update:
                self.io_manager.update_output(**logic_update)

    def get_metadata(self) -> Dict[str, Any]:
        """Aggregate metadata from all managers."""
        return {
            "config": self.config_manager.get_metadata(),
            "engine": {"running": self.running},
            "io": self.io_manager.get_metadata(),
            "logic": self.logic_manager.get_metadata(),
        }

    def _session_prefix(self) -> str:
        """Return '{session_id}_' if set, else ''."""
        session_cfg = self.config_manager.session_config
        if session_cfg is None or not session_cfg.session_id:
            return ""
        return f"{session_cfg.session_id}_"

    def save_md(self):
        """Save session metadata to sample_data_dir, gated by save_metadata."""
        session_cfg = self.config_manager.session_config
        if session_cfg is None or session_cfg.session_parameters is None:
            return
        if not session_cfg.session_parameters.get("save_metadata", False):
            logger.info("save_metadata is false, skipping metadata save")
            return
        metadata = self.get_metadata()
        data_dir = Path(session_cfg.sample_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{self._session_prefix()}metadata.json"
        _save_metadata_to_file(path, metadata)

    def save_data(self, **kwargs):
        """Save data from IO and logic managers."""
        session_cfg = self.config_manager.session_config
        if session_cfg is None or session_cfg.session_parameters is None:
            save_samples = False
        else:
            save_samples = session_cfg.session_parameters.get("save_samples", False)
        if not save_samples:
            logger.info("save_samples is false, skipping data save")
            return
        prefix = self._session_prefix()
        self.io_manager.save_data(prefix=prefix, **kwargs)
        self.logic_manager.save_data(prefix=prefix, **kwargs)

    def close(self):
        """Close all managers."""
        self.running = False
        self.logic_manager.close()
        self.io_manager.close()
