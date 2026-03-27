"""
ClosedLoopEngine for CLEF2.

Orchestrates the real-time closed loop:
  update_input -> process_sample -> check_logic -> update_output
"""

import logging
from typing import Any, Dict, Optional

from clef2.core.config.config_manager import ConfigManager
from clef2.core.io.io_manager import IOManager
from clef2.core.logic.logic_manager import LogicManager

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

        Each iteration:
        1. update_input — data interfaces poll devices and store samples
        2. process_sample — distribute input_stores to all logic algorithms
        3. check_logic — collect logic_update dicts from algorithms
        4. update_output — route logic updates to output devices
        """
        self.running = True
        for _ in range(iterations):
            if not self.running:
                break

            self.io_manager.update_input()
            self.logic_manager.process_sample(self.io_manager.input_stores)

            logic_update = self.logic_manager.check_logic()
            if logic_update:
                self.io_manager.update_output(**logic_update)

    def get_metadata(self) -> Dict[str, Any]:
        """Aggregate metadata from all managers."""
        return {
            "engine": {"running": self.running},
            "io": self.io_manager.get_metadata(),
            "logic": self.logic_manager.get_metadata(),
        }

    def save_data(self, **kwargs):
        """Save data from IO and logic managers."""
        save_samples = self.config_manager.session_config.session_parameters.get(
            "save_samples", False
        )
        if not save_samples:
            logger.info("save_samples is false, skipping data save")
            return
        self.io_manager.save_data(**kwargs)
        self.logic_manager.save_data(**kwargs)

    def close(self):
        """Close all managers."""
        self.running = False
        self.logic_manager.close()
        self.io_manager.close()
