"""
Ring Attractor Demo Hardware Backend - Dual Limit Cycle Version

Simulates a camera observing a ring attractor system with dual concentric rings
using continuous dynamical systems with stable limit cycles.

The backend:
- Generates 100x100 uint16 images with a Gaussian blob puncta
- Maintains state: [x, y] in Cartesian coordinates
- Uses radial dynamics: dr/dt = -k(r-r1)(r-r2) + perturbation
- Uses angular drift: dtheta/dt = omega
- Responds to stimulation by perturbing radially to switch rings
"""

import logging
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple

from hardware.backends.dummy_backend import (
    DummyHardwareBackend,
    DummyCamera,
    DummyStage,
    DummyStimulus
)
from hardware.camera_interface import CameraInterface
from config.config_manager import HardwareConfig

logger = logging.getLogger(__name__)


class RingAttractorDynamics:
    """
    Ring attractor with dual stable limit cycles.
    
    Implements continuous dynamics:
    - dr/dt = -k(r-r1)(r-r2) + u(t)  [radial dynamics with two stable cycles]
    - dtheta/dt = omega               [constant angular drift]
    
    Where r1 and r2 are the radii of the two stable limit cycles.
    """
    
    def __init__(
        self,
        inner_radius: float = 3.0,
        outer_radius: float = 6.0,
        k_radial: float = 5.0,
        omega: float = 1.0,
        dt: float = 0.1,
        image_center: Tuple[float, float] = (50.0, 50.0)
    ):
        """
        Initialize ring attractor system.
        
        Args:
            inner_radius: Radius of inner stable limit cycle (r1)
            outer_radius: Radius of outer stable limit cycle (r2)
            k_radial: Stiffness of radial dynamics (larger = faster attraction)
            omega: Angular velocity (radians per time step)
            dt: Integration time step
            image_center: (cx, cy) center of image in pixels
        """
        self.inner_radius = inner_radius  # r1
        self.outer_radius = outer_radius  # r2
        self.middle_radius = (outer_radius + inner_radius) / 2
        self.k_radial = k_radial
        self.omega = omega
        self.dt = dt
        self.cx, self.cy = image_center
        
        # State variables (Cartesian coordinates relative to center)
        self.x = inner_radius  # Start on inner ring
        self.y = 0.0
        
        # Perturbation signal
        self.perturbation = 0.0
        
        # Track which ring we're on
        self.ring_index = 0  # 0=inner, 1=outer
        
        logger.info(
            f"Ring attractor initialized: r1={inner_radius}, r2={outer_radius}, "
            f"k={k_radial}, omega={omega}, dt={dt}"
        )
    
    def _dynamics(self, x: float, y: float, u: float) -> Tuple[float, float]:
        """
        Compute derivatives dx/dt, dy/dt.
        
        Args:
            x, y: Current Cartesian coordinates
            u: External perturbation (scalar)
            
        Returns:
            (dx, dy) derivatives
        """
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Prevent division by zero
        if r < 1e-6:
            r = 1e-6
            theta = 0.0
        
        # Radial dynamics: dr/dt = -k(r-r1)(r-r2) + u
        # dr = -self.k_radial * (r - self.inner_radius) * (r - self.outer_radius) + u # oops need an unstable orbit separating the two
        dr = -self.k_radial * (r - self.inner_radius) * (r - self.middle_radius) * (r - self.outer_radius) + u
        
        # Angular drift: dtheta/dt = omega
        dtheta = self.omega
        
        # Convert to Cartesian derivatives
        dx = dr * np.cos(theta) - r * dtheta * np.sin(theta)
        dy = dr * np.sin(theta) + r * dtheta * np.cos(theta)
        
        return dx, dy
    
    def step(self) -> Tuple[float, int]:
        """
        Advance dynamics by one time step using Euler integration.
        
        Returns:
            Tuple of (theta, ring_index)
        """
        # Compute derivatives
        dx, dy = self._dynamics(self.x, self.y, self.perturbation)
        
        # Euler step
        self.x += dx * self.dt
        self.y += dy * self.dt
        
        # Decay perturbation
        self.perturbation *= 0.95  # exponential decay
        
        # Update ring classification based on current radius
        r = np.sqrt(self.x**2 + self.y**2)
        self.ring_index = 0 if r < self.middle_radius else 1
        
        # Get current angle
        theta = np.arctan2(self.y, self.x)
        if theta < 0:
            theta += 2 * np.pi
        
        return theta, self.ring_index
    
    def apply_perturbation(self, perturbation_strength: float) -> None:
        """
        Apply radial perturbation with specified strength.
        
        Args:
            perturbation_strength: Perturbation magnitude (-30 to +30)
                Positive = push outward
                Negative = pull inward
        """
        r = np.sqrt(self.x**2 + self.y**2)
        self.perturbation = perturbation_strength
        
        direction = "outward" if perturbation_strength > 0 else "inward"
        logger.debug(f"Applied {direction} perturbation of {perturbation_strength:.1f} at r={r:.1f}")
    
    def toggle_ring(self) -> None:
        """
        Toggle between inner and outer ring by applying strong radial perturbation.
        
        If on inner ring, push outward to outer ring.
        If on outer ring, pull inward to inner ring.
        """
        r = np.sqrt(self.x**2 + self.y**2)
        
        # Determine perturbation direction
        if r < (self.inner_radius + self.outer_radius) / 2.0:
            # On inner ring, push out
            self.perturbation = 15.0  # Strong outward push
            logger.debug(f"Pushing from inner (r={r:.1f}) to outer ring")
        else:
            # On outer ring, pull in
            self.perturbation = -15.0  # Strong inward pull
            logger.debug(f"Pulling from outer (r={r:.1f}) to inner ring")
    
    def perturb_theta(self, delta_theta: float) -> None:
        """
        Perturb angular position by rotating the state vector.
        
        Args:
            delta_theta: Angle to rotate by (radians)
        """
        # Get current radius
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        
        # Apply angular perturbation
        theta_new = theta + delta_theta
        
        # Convert back to Cartesian
        self.x = r * np.cos(theta_new)
        self.y = r * np.sin(theta_new)
        
        logger.debug(f"Angular perturbation: delta_theta={delta_theta:.3f}")
    
    def get_state(self) -> Tuple[float, int]:
        """Get current state (theta, ring_index)."""
        theta = np.arctan2(self.y, self.x)
        if theta < 0:
            theta += 2 * np.pi
        return theta, self.ring_index
    
    def set_state(self, theta: float, ring_index: int) -> None:
        """
        Set state by placing on specified ring at given angle.
        
        Args:
            theta: Angular position
            ring_index: 0 for inner, 1 for outer
        """
        r = self.inner_radius if ring_index == 0 else self.outer_radius
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)
        self.ring_index = ring_index
        self.perturbation = 0.0
    
    def get_cartesian_position(self) -> Tuple[float, float]:
        """
        Get current position in image pixel coordinates.
        
        Returns:
            (x, y) in pixels relative to center
        """
        return self.x, self.y


class RingCamera(DummyCamera):
    """Camera that generates images with puncta on ring attractors."""
    
    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        ring_dynamics: Optional[RingAttractorDynamics] = None,
        noise_level: float = 100.0,
        puncta_brightness: float = 50000.0,
        puncta_radius: float = 3.0,
        exposure_ms: float = 10.0
    ):
        """
        Initialize ring camera.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            ring_dynamics: RingAttractorDynamics instance
            noise_level: Standard deviation of Gaussian noise
            puncta_brightness: Peak brightness of puncta
            puncta_radius: Gaussian sigma for puncta (pixels)
            exposure_ms: Exposure time in milliseconds
        """
        super().__init__(width=width, height=height, input_file=None)
        
        self.ring = ring_dynamics or RingAttractorDynamics()
        self.noise_level = noise_level
        self.puncta_brightness = puncta_brightness
        self.puncta_radius = puncta_radius
        self.exposure_ms = exposure_ms
        
        # Image center
        self.center_x = width / 2.0
        self.center_y = height / 2.0
        
        logger.info(
            f"RingCamera initialized: {width}x{height}, "
            f"noise={noise_level}, brightness={puncta_brightness}, "
            f"puncta_radius={puncta_radius}"
        )
    
    def _draw_puncta(
        self, 
        image: np.ndarray, 
        x: float, 
        y: float
    ) -> None:
        """
        Draw Gaussian puncta at position.
        
        Args:
            image: Image array to modify
            x: X coordinate relative to center
            y: Y coordinate relative to center
        """
        # Convert to pixel coordinates
        px = self.center_x + x
        py = self.center_y + y
        
        # Create coordinate grids
        xx, yy = np.meshgrid(
            np.arange(self.width),
            np.arange(self.height)
        )
        
        # Gaussian blob
        dist_sq = (xx - px)**2 + (yy - py)**2
        gaussian = self.puncta_brightness * np.exp(-dist_sq / (2 * self.puncta_radius**2))
        
        image += gaussian
    
    def get_image(self) -> np.ndarray:
        """
        Generate image with puncta at current ring position.
        
        Returns:
            Image array with puncta on ring
        """
        # Advance dynamics
        theta, ring_idx = self.ring.step()
        
        # Create base image with noise
        image = np.random.normal(0, self.noise_level, (self.height, self.width))
        image = np.clip(image, 0, None).astype(np.float32)
        
        # Get puncta position
        x, y = self.ring.get_cartesian_position()
        
        # Draw puncta
        self._draw_puncta(image, x, y)
        
        # Convert to uint16 and clip
        image = np.clip(image, 0, 65535).astype(np.uint16)
        
        # Sleep for exposure time
        time.sleep(self.exposure_ms / 1000.0)
        
        return image


class RingStimulus(DummyStimulus):
    """Stimulus that toggles ring and perturbs theta."""
    
    def __init__(self, ring_dynamics: RingAttractorDynamics, **kwargs):
        """
        Initialize ring stimulus.
        
        Args:
            ring_dynamics: RingAttractorDynamics instance to perturb
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.ring = ring_dynamics
    
    def activate_stimulus(self, params: Dict[str, Any]) -> None:
        """
        Apply radial perturbation with specified intensity.
        
        Args:
            params: Dictionary with 'intensity' (-30 to +30)
                Positive = push outward, Negative = pull inward
        """
        super().activate_stimulus(params)
        
        # Get perturbation strength from intensity parameter
        perturbation_strength = params.get('intensity', 0.0)
        
        # Apply radial perturbation
        self.ring.apply_perturbation(perturbation_strength)
        
        # Optional: Add small angular perturbation (10% of intensity)
        # delta_theta = np.random.normal(0, abs(perturbation_strength) * 0.01)
        # self.ring.perturb_theta(delta_theta)
        
        logger.info(
            f"Applied ring stimulus: perturbation={perturbation_strength:.1f}, "
            # f"delta_theta={delta_theta:.3f}"
        )


class RingAttractorBackend(DummyHardwareBackend):
    """
    Hardware backend for ring attractor demonstration.
    
    Generates images with a puncta orbiting one of two concentric rings.
    Stimulus toggles between rings and perturbs angular position.
    """
    
    def __init__(self, config: HardwareConfig):
        """Initialize ring attractor backend."""
        super().__init__(config)
        self.ring_dynamics = None
    
    def initialize(self, **kwargs) -> None:
        """
        Initialize ring attractor hardware.
        
        Uses config.ring_params for all parameters.
        """
        logger.info("Initializing ring attractor backend...")
        
        # Extract ring parameters from config
        ring_params = self.config.ring_params
        width = ring_params.get('image_width', 100)
        height = ring_params.get('image_height', 100)
        
        # Create ring dynamics
        self.ring_dynamics = RingAttractorDynamics(
            inner_radius=ring_params.get('inner_radius', 1),
            outer_radius=ring_params.get('outer_radius', 3),
            k_radial=ring_params.get('k_radial', 5.0),
            omega=ring_params.get('omega', 1.0),
            dt=ring_params.get('dt', 0.001),
            image_center=(width / 2.0, height / 2.0)
        )
        
        # Set initial state if provided
        initial_state = ring_params.get('initial_state')
        if initial_state is not None:
            theta, ring_idx = initial_state
            self.ring_dynamics.set_state(theta, ring_idx)
        
        # Create ring camera
        self._camera = RingCamera(
            width=width,
            height=height,
            ring_dynamics=self.ring_dynamics,
            noise_level=ring_params.get('noise_level', 100.0),
            puncta_brightness=ring_params.get('puncta_brightness', 5000.0),
            puncta_radius=ring_params.get('puncta_radius', 3.0),
            exposure_ms=ring_params.get('exposure_ms', 10.0)
        )
        
        # Set exposure
        self._camera.set_exposure(ring_params.get('exposure_ms', 10.0))
        
        # Create dummy stage
        self._stage = DummyStage()
        
        # Create ring stimulus
        self._stimulus = RingStimulus(ring_dynamics=self.ring_dynamics)
        
        self._initialized = True
        logger.info("Ring attractor backend initialized successfully")
        theta, ring_idx = self.ring_dynamics.get_state()
        logger.info(f"Initial state: theta={theta:.3f}, ring={ring_idx}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get hardware metadata including ring parameters."""
        metadata = super().get_metadata()
        
        if self.ring_dynamics:
            theta, ring_idx = self.ring_dynamics.get_state()
            metadata.update({
                'ring_inner_radius': self.ring_dynamics.inner_radius,
                'ring_outer_radius': self.ring_dynamics.outer_radius,
                'ring_k_radial': self.ring_dynamics.k_radial,
                'ring_omega': self.ring_dynamics.omega,
                'ring_dt': self.ring_dynamics.dt,
                'ring_final_theta': theta,
                'ring_final_ring_index': ring_idx,
                'ring_final_position': [self.ring_dynamics.x, self.ring_dynamics.y],
            })
        
        return metadata
