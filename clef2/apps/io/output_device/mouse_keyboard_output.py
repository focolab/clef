"""
Mouse and Keyboard Output Device for CLEF2.

Delivers keyboard presses and mouse clicks/movements via pynput.
Supports smooth arc trajectories for mouse movement.
"""

import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import splprep, splev

from clef2.core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)


class MouseKeyboardOutput(BaseOutputDevice):
    """Output device for keyboard and mouse input simulation via pynput."""

    device_class: ClassVar[Optional[str]] = "mouse_keyboard"
    device_type: ClassVar[Optional[str]] = "software"

    def __init__(
        self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None
    ):
        super().__init__(name, config, io_manager=io_manager)
        self.keyboard = None
        self.mouse = None
        self.Key = None
        self.Button = None
        self.input_events: List[Dict[str, Any]] = []

    def connect(self):
        """Initialize pynput keyboard and mouse controllers."""
        try:
            from pynput.keyboard import Controller as KeyboardController, Key
            from pynput.mouse import Controller as MouseController, Button

            self.keyboard = KeyboardController()
            self.mouse = MouseController()
            self.Key = Key
            self.Button = Button
            logger.info(f"MouseKeyboardOutput '{self.name}' connected to pynput")
        except ImportError:
            logger.error(
                f"MouseKeyboardOutput '{self.name}': pynput not available"
            )

    def configure(self):
        """Read mouse movement parameters from config."""
        cfg = self.config
        self.default_keys = cfg.get("default_key_sequence", ["space"])
        self.default_mouse_button = cfg.get("default_mouse_button", "left")
        self.mouse_curvature = cfg.get("mouse_curvature", 0.3)
        self.mouse_overshoot = cfg.get("mouse_overshoot", 0.2)
        self.mouse_randomness = cfg.get("mouse_randomness", 0.9)
        self.mouse_arc_points = cfg.get("mouse_arc_points", 20)
        self.mouse_scaling_factor = cfg.get("mouse_scaling_factor", 1.0)
        self.mouse_step_delay = cfg.get("mouse_step_delay", 0.04)
        self.pre_click_delay_range = cfg.get("pre_click_delay_range", [0.2, 1.5])
        logger.info(
            f"MouseKeyboardOutput '{self.name}' configured: "
            f"curvature={self.mouse_curvature}, overshoot={self.mouse_overshoot}"
        )

    def _update_output(self, **kwargs):
        """Execute keyboard and/or mouse actions.

        Keyword Args:
            keys: list of key name strings to press sequentially.
            click_position: (x, y) tuple to click at.
            mouse_button: "left", "right", or "middle".
            smooth: bool, use arc trajectory (default True). False = instant move.
        """
        keys = kwargs.get("keys")
        click_position = kwargs.get("click_position")

        if keys:
            self._press_keys(keys)

        if click_position is not None:
            smooth = kwargs.get("smooth", True)
            button_name = kwargs.get("mouse_button", self.default_mouse_button)
            self._click_at(click_position, button_name, smooth)

    # -- keyboard ---------------------------------------------------------

    def _parse_key(self, key_name: str):
        """Parse a key name string to a pynput Key or character."""
        key_name = key_name.lower()
        if hasattr(self.Key, key_name):
            return getattr(self.Key, key_name)
        if len(key_name) == 1:
            return key_name
        logger.warning(f"Unknown key: {key_name}")
        return None

    def _press_keys(self, keys: List[str]):
        """Press and release a sequence of keys."""
        if not self.keyboard:
            return
        for name in keys:
            key = self._parse_key(name)
            if key is None:
                continue
            try:
                self.keyboard.press(key)
                time.sleep(0.01)
                self.keyboard.release(key)
                self.input_events.append(
                    {"type": "keyboard", "key": str(key), "time": time.time()}
                )
            except Exception as e:
                logger.error(f"Error pressing key {name}: {e}")

    # -- mouse ------------------------------------------------------------

    def _get_button(self, name: str):
        """Parse button name to pynput Button."""
        if name == "right":
            return self.Button.right
        if name == "middle":
            return self.Button.middle
        return self.Button.left

    def _click_at(
        self,
        position: Tuple[float, float],
        button_name: str = "left",
        smooth: bool = True,
    ):
        """Move mouse to position and click."""
        if not self.mouse:
            return

        p1 = (float(position[0]), float(position[1]))

        if smooth:
            p0 = self.mouse.position
            arc = self._make_arc(
                p0,
                p1,
                curvature=self.mouse_curvature,
                randomness=self.mouse_randomness,
                n_points=self.mouse_arc_points,
                overshoot=self.mouse_overshoot,
            )
            for t in range(arc.shape[0]):
                self.mouse.position = (
                    arc[t, 0] * self.mouse_scaling_factor,
                    arc[t, 1] * self.mouse_scaling_factor,
                )
                time.sleep(self.mouse_step_delay)
        else:
            self.mouse.position = (
                p1[0] * self.mouse_scaling_factor,
                p1[1] * self.mouse_scaling_factor,
            )

        # Pre-click delay
        lo, hi = self.pre_click_delay_range
        time.sleep(np.random.uniform(lo, hi))

        button = self._get_button(button_name)
        self.mouse.click(button, 1)

        self.input_events.append(
            {
                "type": "mouse_click",
                "position": p1,
                "button": button_name,
                "smooth": smooth,
                "time": time.time(),
            }
        )
        logger.info(f"Clicked {button_name} at ({p1[0]:.0f}, {p1[1]:.0f})")

    # -- arc trajectory ---------------------------------------------------

    def _make_arc(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        n_points: int = 20,
        curvature: float = 0.3,
        randomness: float = 0.0,
        overshoot: float = 0.2,
    ) -> np.ndarray:
        """Smooth spline arc from p0 to p1. Returns (n_points, 2) array."""
        p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
        direction = p1 - p0
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            return np.tile(p1, (n_points, 1))

        perp = np.array([-direction[1], direction[0]]) / dist

        mid1 = p0 + 0.3 * direction + curvature * dist * perp
        mid2 = p0 + 0.7 * direction - curvature * dist * perp
        overshoot_pt = p1 + overshoot * direction

        mid1 += randomness * np.random.randn(2)
        mid2 += randomness * np.random.randn(2)
        overshoot_pt += randomness * np.random.randn(2)

        x = [p0[0], mid1[0], mid2[0], overshoot_pt[0], p1[0]]
        y = [p0[1], mid1[1], mid2[1], overshoot_pt[1], p1[1]]

        tck, _ = splprep([x, y], s=0, k=2)
        u = np.linspace(0, 1, n_points)
        x_s, y_s = splev(u, tck)
        return np.column_stack([x_s, y_s])

    # -- metadata / close -------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        base = super().get_metadata()
        base.update(
            {
                "input_events": self.input_events,
                "num_input_events": len(self.input_events),
            }
        )
        return base

    def close(self):
        logger.info(
            f"MouseKeyboardOutput '{self.name}' closed. "
            f"Delivered {len(self.input_events)} events"
        )
