"""
Lorenz Attractor Demo Hardware Backend

Simulates a camera observing a Lorenz attractor system encoded as 3 bright pixels.
Each pixel's position encodes one dimension (x, y, z) of the Lorenz state.

The backend:
- Generates 20x20 images with 3 bright pixels
- Advances Lorenz dynamics between frames using Runge-Kutta integration
- Adds configurable noise
- Responds to stimulation by perturbing the state variables
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
from config.config_manager import HardwareConfig

logger = logging.getLogger(__name__)


class LorenzDynamics:
    """Lorenz attractor dynamics with RK4 integration."""
    
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
        """
        Initialize Lorenz system.
        
        Args:
            sigma: Prandtl number (default 10.0)
            rho: Rayleigh number (default 28.0)
            beta: Geometric factor (default 8/3)
            dt: Time step for integration
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        
        # State vector [x, y, z]
        self.state = np.array([1.0, 1.0, 1.0])
        
        logger.info(f"Lorenz system initialized: σ={sigma}, ρ={rho}, β={beta:.3f}, dt={dt}")
    
    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Lorenz system derivatives.
        
        Args:
            state: Current state [x, y, z]
            
        Returns:
            Time derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])
    
    def step_rk4(self) -> np.ndarray:
        """
        Advance state by one time step using RK4 integration.
        
        Returns:
            New state [x, y, z]
        """
        # Fourth-order Runge-Kutta
        k1 = self.derivatives(self.state)
        k2 = self.derivatives(self.state + 0.5 * self.dt * k1)
        k3 = self.derivatives(self.state + 0.5 * self.dt * k2)
        k4 = self.derivatives(self.state + self.dt * k3)
        
        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.state.copy()
    
    def perturb_state(self, perturbation: np.ndarray) -> None:
        """
        Apply perturbation to current state (stimulus effect).
        
        Args:
            perturbation: Vector [dx, dy, dz] to add to state
        """
        self.state += perturbation
        logger.debug(f"Lorenz state perturbed by {perturbation}, new state: {self.state}")
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()
    
    def set_state(self, state: np.ndarray) -> None:
        """Set state vector."""
        self.state = state.copy()


class LorenzCamera(DummyCamera):
    """Camera that generates images encoding Lorenz attractor state."""
    
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        lorenz: Optional[LorenzDynamics] = None,
        noise_level: float = 100.0,
        pixel_brightness: float = 50000.0,
        pixel_radius: int = 2
    ):
        """
        Initialize Lorenz camera.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            lorenz: LorenzDynamics instance (created if None)
            noise_level: Standard deviation of Gaussian noise
            pixel_brightness: Brightness of encoding pixels
            pixel_radius: Radius of bright pixels
        """
        super().__init__(width=width, height=height, input_file=None)
        
        self.lorenz = lorenz or LorenzDynamics()
        self.noise_level = noise_level
        self.pixel_brightness = pixel_brightness
        self.pixel_radius = pixel_radius
        
        # Scaling factors to map Lorenz coordinates to image space
        # Lorenz attractor roughly spans: x∈[-20,20], y∈[-30,30], z∈[0,50]
        self.x_scale = (width - 2 * pixel_radius - 1) / 40.0  # span of 40
        self.x_offset = 20.0  # center at 0
        
        self.y_scale = (height - 2 * pixel_radius - 1) / 60.0  # span of 60
        self.y_offset = 30.0  # center at 0
        
        # For z, we'll use a separate pixel brightness or position
        # Let's encode z as vertical position in right half of image
        self.z_scale = (height - 2 * pixel_radius - 1) / 50.0  # z spans [0, 50]
        
        logger.info(
            f"LorenzCamera initialized: {width}x{height}, "
            f"noise={noise_level}, brightness={pixel_brightness}"
        )
    
    def _lorenz_to_pixel_coords(self, state: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Convert Lorenz state to pixel coordinates.
        
        Strategy: 
        - X coordinate → horizontal position in left half
        - Y coordinate → vertical position in left half  
        - Z coordinate → vertical position in right half
        
        Args:
            state: Lorenz state [x, y, z]
            
        Returns:
            Three (row, col) tuples for the three encoding pixels
        """
        x, y, z = state
        
        # X pixel: left side, vertical position based on x value
        x_col = self.pixel_radius  # left edge
        x_row = int((x + self.x_offset) * self.x_scale) + self.pixel_radius
        x_row = np.clip(x_row, self.pixel_radius, self.height - self.pixel_radius - 1)
        
        # Y pixel: middle-left, vertical position based on y value
        y_col = self.width // 3
        y_row = int((y + self.y_offset) * self.y_scale) + self.pixel_radius
        y_row = np.clip(y_row, self.pixel_radius, self.height - self.pixel_radius - 1)
        
        # Z pixel: right side, vertical position based on z value
        z_col = 2 * self.width // 3
        z_row = int(z * self.z_scale) + self.pixel_radius
        z_row = np.clip(z_row, self.pixel_radius, self.height - self.pixel_radius - 1)
        
        return (x_row, x_col), (y_row, y_col), (z_row, z_col)
    
    def _draw_bright_pixel(self, image: np.ndarray, row: int, col: int) -> None:
        """
        Draw a bright pixel with Gaussian profile.
        
        Args:
            image: Image array to modify
            row: Center row
            col: Center column
        """
        r = self.pixel_radius
        for dr in range(-r, r+1):
            for dc in range(-r, r+1):
                r_pos = row + dr
                c_pos = col + dc
                if 0 <= r_pos < self.height and 0 <= c_pos < self.width:
                    # Gaussian falloff
                    dist = np.sqrt(dr**2 + dc**2)
                    intensity = self.pixel_brightness * np.exp(-dist**2 / (2 * (r/2)**2))
                    image[r_pos, c_pos] += intensity
    
    def get_image(self) -> np.ndarray:
        """
        Generate image encoding current Lorenz state.
        
        Returns:
            Image array with 3 bright pixels
        """
        # Advance Lorenz dynamics
        state = self.lorenz.step_rk4()
        
        # Create base image with noise
        image = np.random.normal(0, self.noise_level, (self.height, self.width))
        image = np.clip(image, 0, None).astype(np.float32)
        
        # Get pixel positions for current state
        x_pos, y_pos, z_pos = self._lorenz_to_pixel_coords(state)
        
        # Draw three bright pixels
        self._draw_bright_pixel(image, *x_pos)
        self._draw_bright_pixel(image, *y_pos)
        self._draw_bright_pixel(image, *z_pos)
        
        # Convert to uint16 and clip
        image = np.clip(image, 0, 65535).astype(np.uint16)
        
        # Sleep for exposure time
        time.sleep(self.exposure_ms / 1000.0)
        
        return image


class LorenzStimulus(DummyStimulus):
    """Stimulus that perturbs Lorenz state."""
    
    def __init__(self, lorenz: LorenzDynamics, **kwargs):
        """
        Initialize Lorenz stimulus.
        
        Args:
            lorenz: LorenzDynamics instance to perturb
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.lorenz = lorenz
        self.perturbation = kwargs.get('perturbation', [2.0, 2.0, 2.0])
    
    def activate_stimulus(self, params: Dict[str, Any]) -> None:
        """
        Apply perturbation to Lorenz state.
        
        Args:
            params: Dictionary with optional 'perturbation' key
        """

        super().activate_stimulus(params)
        
        # Scale by intensity if provided
        intensity_scale = params.get('intensity', 10)
        perturbation = params.get('perturbation', [2.0, 2.0, 2.0])
        perturbation = np.array(perturbation) * intensity_scale
        
        # Apply to Lorenz system
        self.lorenz.perturb_state(perturbation)
        
        logger.info(f"Applied Lorenz stimulus: params={params}, perturbation{perturbation}")



class LorenzDemoBackend(DummyHardwareBackend):
    """
    Hardware backend for Lorenz attractor demonstration.
    
    Generates images encoding Lorenz attractor dynamics as 3 bright pixels.
    Stimulus perturbs the Lorenz state variables.
    """
    
    def __init__(self, config: HardwareConfig):
        """Initialize Lorenz demo backend."""
        super().__init__(config)
        
        # Lorenz system parameters (will be set from config in initialize)
        self.lorenz = None
    
    def initialize(self, **kwargs) -> None:
        """
        Initialize Lorenz demo hardware.
        
        Args:
            **kwargs: Configuration including:
                - sigma: Lorenz sigma parameter (default 10.0)
                - rho: Lorenz rho parameter (default 28.0)
                - beta: Lorenz beta parameter (default 8/3)
                - dt: Integration time step (default 0.01)
                - noise_level: Image noise std (default 100.0)
                - pixel_brightness: Brightness of encoding pixels (default 50000.0)
                - pixel_radius: Radius of bright pixels (default 2)
                - image_width: Image width (default 20)
                - image_height: Image height (default 20)
                - initial_state: Initial [x,y,z] state (default [1,1,1])
        """
        logger.info("Initializing Lorenz demo backend...")
        
        # Extract Lorenz parameters
        sigma = kwargs.get('sigma', 10.0)
        rho = kwargs.get('rho', 28.0)
        beta = kwargs.get('beta', 8.0/3.0)
        dt = kwargs.get('dt', 0.01)
        
        # Create Lorenz dynamics
        self.lorenz = LorenzDynamics(sigma=sigma, rho=rho, beta=beta, dt=dt)
        
        # Set initial state if provided
        # initial_state = kwargs.get('initial_state')
        initial_state = self.config.lorenz_params.get('initial_state')
        if initial_state is not None:
            self.lorenz.set_state(np.array(initial_state))
        
        # Extract camera parameters
        width = self.config.lorenz_params.get('image_width')
        height = self.config.lorenz_params.get('image_width')
        noise_level = self.config.lorenz_params.get('noise_level')
        pixel_brightness = kwargs.get('pixel_brightness', 50000.0)
        pixel_radius = kwargs.get('pixel_radius', 2)
        
        # Create Lorenz camera
        self._camera = LorenzCamera(
            width=width,
            height=height,
            lorenz=self.lorenz,
            noise_level=noise_level,
            pixel_brightness=pixel_brightness,
            pixel_radius=pixel_radius
        )
        
        # Create dummy stage
        self._stage = DummyStage()
        
        # Create Lorenz stimulus
        self._stimulus = LorenzStimulus(lorenz=self.lorenz)
        
        self._initialized = True
        logger.info("Lorenz demo backend initialized successfully")
        logger.info(f"Initial Lorenz state: {self.lorenz.get_state()}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get hardware metadata including Lorenz parameters."""
        metadata = super().get_metadata()
        
        if self.lorenz:
            metadata.update({
                'lorenz_sigma': self.lorenz.sigma,
                'lorenz_rho': self.lorenz.rho,
                'lorenz_beta': self.lorenz.beta,
                'lorenz_dt': self.lorenz.dt,
                'lorenz_final_state': self.lorenz.get_state().tolist(),
            })
        
        return metadata
