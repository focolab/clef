"""
Ring Attractor Output Device for CLEF2.

Applies radial and angular perturbations to the ring attractor dynamics
owned by the linked input device.
"""

import logging
from typing import Any, ClassVar, Dict, Optional

from core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)


class RingAttractorOutputDevice(BaseOutputDevice):
    """Output device that perturbs ring attractor dynamics via a linked input device."""

    device_class: ClassVar[Optional[str]] = "ring_attractor_output"
    device_type: ClassVar[Optional[str]] = "demo"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self._dynamics = None

    def _resolve_dynamics(self):
        """Lazy lookup of the linked input device's dynamics on first use."""
        linked_name = self.config.get("linked_input_device")
        if linked_name is None:
            raise ValueError(
                f"RingAttractorOutputDevice '{self.name}' requires "
                f"'linked_input_device' in output_device_parameters"
            )
        input_dev = self.io_manager.get_input_device(linked_name)
        self._dynamics = input_dev.dynamics
        logger.info(f"Output device '{self.name}' linked to input device '{linked_name}'")

    def _update_output(self, **kwargs):
        """Apply radial and angular perturbation to the ring attractor dynamics."""
        if self._dynamics is None:
            self._resolve_dynamics()

        radial = kwargs.get("radial_perturbation", 0.0)
        omega = kwargs.get("omega_perturbation", 0.0)
        self._dynamics.apply_perturbation(radial, omega)
