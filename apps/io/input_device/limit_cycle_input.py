"""
Limit Cycle Input Device for CLEF.

Simulates a camera observing a polar-coordinate dynamical system with dual
concentric stable limit cycles. Generates 100x100 uint16 images with a
Gaussian blob puncta whose position is governed by the continuous dynamics.
"""

import logging
import time
import numpy as np
from typing import Any, ClassVar, Dict, Optional, Tuple

from core.io.input_device.BaseInputDevice import BaseInputDevice

logger = logging.getLogger(__name__)


class LimitCycleDynamics:
    """
    Polar-coordinate dynamics with dual stable limit cycles.

    Implements continuous dynamics:
    - dr/dt = -k(r-r1)(r-r_mid)(r-r2) + u_r(t)
    - dtheta/dt = omega + u_omega(t)
    """

    def __init__(
        self,
        inner_radius: float = 3.0,
        outer_radius: float = 6.0,
        k_radial: float = 5.0,
        omega: float = 1.0,
        dt: float = 0.1,
        image_center: Tuple[float, float] = (50.0, 50.0),
    ):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.middle_radius = (outer_radius + inner_radius) / 2
        self.k_radial = k_radial
        self.omega = omega
        self.dt = dt
        self.cx, self.cy = image_center

        # State variables (Cartesian coordinates relative to center)
        self.x = inner_radius
        self.y = 0.0

        # Perturbation signals
        self.perturbation = 0.0
        self.omega_perturbation = 0.0

        # Track which ring we're on
        self.ring_index = 0  # 0=inner, 1=outer

        logger.info(
            f"Limit cycle dynamics initialized: r1={inner_radius}, r2={outer_radius}, "
            f"k={k_radial}, omega={omega}, dt={dt}"
        )

    def _dynamics(self, x: float, y: float, u_r: float, u_omega: float) -> Tuple[float, float]:
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        if r < 1e-6:
            r = 1e-6
            theta = 0.0

        dr = -self.k_radial * (r - self.inner_radius) * (r - self.middle_radius) * (r - self.outer_radius) + u_r
        dtheta = self.omega + u_omega

        dx = dr * np.cos(theta) - r * dtheta * np.sin(theta)
        dy = dr * np.sin(theta) + r * dtheta * np.cos(theta)

        return dx, dy

    def step(self) -> Tuple[float, int]:
        dx, dy = self._dynamics(self.x, self.y, self.perturbation, self.omega_perturbation)

        self.x += dx * self.dt
        self.y += dy * self.dt

        self.perturbation *= 0.95
        self.omega_perturbation *= 0.95

        r = np.sqrt(self.x**2 + self.y**2)
        self.ring_index = 0 if r < self.middle_radius else 1

        theta = np.arctan2(self.y, self.x)
        if theta < 0:
            theta += 2 * np.pi

        return theta, self.ring_index

    def apply_perturbation(self, perturbation_strength: float, omega_perturbation: float = 0.0) -> None:
        self.perturbation = perturbation_strength
        self.omega_perturbation = omega_perturbation
        logger.debug(
            f"Applied perturbation: radial={perturbation_strength:.1f}, "
            f"omega={omega_perturbation:.2f}"
        )

    def get_state(self) -> Tuple[float, int]:
        theta = np.arctan2(self.y, self.x)
        if theta < 0:
            theta += 2 * np.pi
        return theta, self.ring_index

    def set_state(self, theta: float, ring_index: int) -> None:
        r = self.inner_radius if ring_index == 0 else self.outer_radius
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)
        self.ring_index = ring_index
        self.perturbation = 0.0
        self.omega_perturbation = 0.0

    def get_cartesian_position(self) -> Tuple[float, float]:
        return self.x, self.y


class LimitCycleInputDevice(BaseInputDevice):
    """Input device that generates limit cycle images."""

    device_class: ClassVar[Optional[str]] = "limit_cycle_input"
    device_type: ClassVar[Optional[str]] = "demo"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)

        cfg = self.config
        width = cfg.get("image_width", 100)
        height = cfg.get("image_height", 100)

        self.dynamics = LimitCycleDynamics(
            inner_radius=cfg.get("inner_radius", 3.0),
            outer_radius=cfg.get("outer_radius", 6.0),
            k_radial=cfg.get("k_radial", 5.0),
            omega=cfg.get("omega", 1.0),
            dt=cfg.get("dt", 0.1),
            image_center=(width / 2.0, height / 2.0),
        )

        initial_state = cfg.get("initial_state")
        if initial_state is not None:
            self.dynamics.set_state(initial_state[0], initial_state[1])

        self.width = width
        self.height = height
        self.noise_level = cfg.get("noise_level", 100.0)
        self.puncta_brightness = cfg.get("puncta_brightness", 50000.0)
        self.puncta_radius = cfg.get("puncta_radius", 3.0)
        self.exposure_ms = cfg.get("exposure_ms", 10.0)

        self.center_x = width / 2.0
        self.center_y = height / 2.0

        logger.info(f"LimitCycleInputDevice '{name}' initialized: {width}x{height}")

    def _get_input(self) -> np.ndarray:
        """Advance dynamics and generate a uint16 image with puncta."""
        self.dynamics.step()

        image = np.random.normal(0, max(self.noise_level, 1e-6), (self.height, self.width))
        image = np.clip(image, 0, None).astype(np.float32)

        x, y = self.dynamics.get_cartesian_position()
        px = self.center_x + x
        py = self.center_y + y

        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        dist_sq = (xx - px) ** 2 + (yy - py) ** 2
        gaussian = self.puncta_brightness * np.exp(-dist_sq / (2 * self.puncta_radius**2))
        image += gaussian

        image = np.clip(image, 0, 65535).astype(np.uint16)

        time.sleep(self.exposure_ms / 1000.0)

        return image
