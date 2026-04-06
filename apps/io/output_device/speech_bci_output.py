"""
SpeechBCI Output Device for CLEF.

Applies adaptive streaming parameters to the linked SpeechBCI input device.
Follows the same linked-device pattern as RingAttractorOutputDevice.
"""

import logging
from typing import Any, ClassVar, Dict, Optional

from core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)


class SpeechBCIOutputDevice(BaseOutputDevice):
    """Output device that adjusts streaming parameters on the linked BCI input device."""

    device_class: ClassVar[Optional[str]] = "speech_bci_output"
    device_type: ClassVar[Optional[str]] = "bci"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self._input_device = None

    def _resolve_input_device(self):
        """Lazy lookup of the linked input device on first use."""
        linked_name = self.config.get("linked_input_device")
        if linked_name is None:
            raise ValueError(
                f"SpeechBCIOutputDevice '{self.name}' requires "
                f"'linked_input_device' in output_device_parameters"
            )
        self._input_device = self.io_manager.get_input_device(linked_name)
        logger.info(f"Output device '{self.name}' linked to input device '{linked_name}'")

    def _update_output(self, **kwargs):
        if self._input_device is None:
            self._resolve_input_device()

        # Params that also need to be sent to the cloud server
        cloud_config = {}

        if "decode_interval_bins" in kwargs:
            old = self._input_device.decode_interval_bins
            self._input_device.decode_interval_bins = kwargs["decode_interval_bins"]
            cloud_config["decode_interval_bins"] = kwargs["decode_interval_bins"]
            logger.info(f"decode_interval_bins: {old} -> {kwargs['decode_interval_bins']}")

        if "inter_trial_pause_s" in kwargs:
            old = self._input_device.inter_trial_pause_s
            self._input_device.inter_trial_pause_s = kwargs["inter_trial_pause_s"]
            logger.info(f"inter_trial_pause_s: {old:.1f} -> {kwargs['inter_trial_pause_s']:.1f}")

        if "min_decode_bins" in kwargs:
            old = self._input_device.min_decode_bins
            self._input_device.min_decode_bins = kwargs["min_decode_bins"]
            cloud_config["min_decode_bins"] = kwargs["min_decode_bins"]
            logger.info(f"min_decode_bins: {old} -> {kwargs['min_decode_bins']}")

        # Send config update to cloud server if applicable
        if cloud_config and self._input_device.decoder_mode == "cloud":
            self._input_device.send_cloud_config(cloud_config)
