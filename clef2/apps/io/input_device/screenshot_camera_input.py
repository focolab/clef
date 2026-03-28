"""
Screenshot Camera Input Device for CLEF2.

Captures screen regions as uint8 RGB frames using mss or PIL.
"""

import logging
import numpy as np
from typing import Any, ClassVar, Dict, Optional

from clef2.core.io.input_device.BaseInputDevice import BaseInputDevice

logger = logging.getLogger(__name__)


class ScreenshotCameraInput(BaseInputDevice):
    """Input device that captures screen regions as RGB frames."""

    device_class: ClassVar[Optional[str]] = "screenshot_camera"
    device_type: ClassVar[Optional[str]] = "camera"

    def __init__(
        self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None
    ):
        super().__init__(name, config, io_manager=io_manager)
        self.width = 0
        self.height = 0
        self._capture_func = None
        self._mss = None

    def connect(self):
        """Initialize the capture backend (mss or pil)."""
        backend = self.config.get("backend", "mss")
        if backend == "mss":
            try:
                import mss
                self._mss = mss.mss()
                self._capture_func = self._capture_mss
                logger.info(f"ScreenshotCameraInput '{self.name}' using mss backend")
            except ImportError:
                logger.warning("mss not available, falling back to PIL")
                self._init_pil()
        elif backend == "pil":
            self._init_pil()
        else:
            raise ValueError(f"Unknown screenshot backend: {backend}")

    def _init_pil(self):
        from PIL import ImageGrab
        self._ImageGrab = ImageGrab
        self._capture_func = self._capture_pil
        logger.info(f"ScreenshotCameraInput '{self.name}' using PIL backend")

    def configure(self):
        """Read capture region from config."""
        self.x = self.config.get("x", 0)
        self.y = self.config.get("y", 0)
        self.width = self.config.get("width", 1920)
        self.height = self.config.get("height", 1080)
        self.monitor = self.config.get("monitor", 1)
        logger.info(
            f"ScreenshotCameraInput '{self.name}' configured: "
            f"region=({self.x}, {self.y}, {self.width}, {self.height}), "
            f"monitor={self.monitor}"
        )

    def _capture_mss(self) -> np.ndarray:
        monitor = {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }
        sct_img = self._mss.grab(monitor)
        img = np.array(sct_img)[:, :, :3]  # drop alpha
        img = img[:, :, [2, 1, 0]]  # BGR to RGB
        return img.astype(np.uint8)

    def _capture_pil(self) -> np.ndarray:
        bbox = (self.x, self.y, self.x + self.width, self.y + self.height)
        img = self._ImageGrab.grab(bbox)
        return np.array(img).astype(np.uint8)

    def _get_input(self) -> Optional[np.ndarray]:
        """Capture a screenshot frame. Returns (height, width, 3) uint8 array."""
        return self._capture_func()

    def get_metadata(self) -> Dict[str, Any]:
        base = super().get_metadata()
        base.update({
            "capture_region": {
                "x": self.x, "y": self.y,
                "width": self.width, "height": self.height,
            },
            "color_space": "RGB",
            "bit_depth": 8,
        })
        return base

    def close(self):
        if self._mss is not None:
            self._mss.close()
        logger.info(f"ScreenshotCameraInput '{self.name}' closed")
