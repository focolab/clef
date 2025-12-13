"""
Lorenz Demo Algorithm

Extracts Lorenz attractor state from images and triggers stimulation when
the state enters a configurable 3D volume in phase space.

The algorithm:
- Finds 3 local maxima per frame (encoding x, y, z coordinates)
- Stores timeseries of (x, y, z) values
- Triggers stimulation when state enters defined 3D volume
- Provides real-time 3D state space visualization
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import ndimage

logger = logging.getLogger(__name__)


class LorenzDemoAlgorithm:
    """
    Algorithm for Lorenz attractor closed-loop demo.
    
    Extracts state from images, monitors phase space, and triggers
    stimulation when trajectory enters defined regions.
    """
    
    def __init__(
        self,
        algorithm_config: Optional['AlgorithmConfig'] = None,
        experiment_config: Optional['ExperimentConfig'] = None,
        hardware_manager: Optional['HardwareManager'] = None,
        local_handles: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Lorenz demo algorithm.
        
        Args:
            algorithm_config: Algorithm configuration
            experiment_config: Experiment configuration
            hardware_manager: Hardware manager instance
            local_handles: Dictionary of local handles
            args: Legacy args dictionary
        """
        if args is None:
            args = {}
        if local_handles is None:
            local_handles = {}
        
        self.args = args
        self.local_handles = local_handles
        self.algorithm_config = algorithm_config
        self.experiment_config = experiment_config
        self.hardware_manager = hardware_manager
        
        # Extract config parameters
        gooey_args = self.args.get("gooey_args", {})
        self.frames_to_grab = gooey_args.get("total_frames", 500)
        self.zsize = gooey_args.get("zsize", 1)
        
        # Get algorithm-specific parameters
        if algorithm_config:
            params = algorithm_config.algorithm_params
            
            # Trigger volume definition (box in phase space)
            # Default: trigger when x>0, y>0, z>25 (upper wing)
            self.trigger_volume = {
                'x_min': params.trigger_x_min,
                'x_max': params.trigger_x_max,
                'y_min': params.trigger_y_min,
                'y_max': params.trigger_y_max,
                'z_min': params.trigger_z_min,
                'z_max': params.trigger_z_max,
            }
            
            # Stimulus perturbation vector
            # self.perturbation = self.hardware_manager.config.lorenz_params.get('perturbation')
            self.perturbation = params.perturbation
            
            # Cooldown and stimulus params
            self.stim_cooldown_frames = params.stim_cooldown_frames
            self.stim_duration = algorithm_config.stimulus_params.duration_frames
            self.stim_intensity = algorithm_config.stimulus_params.intensity_percent
        else:
            # Defaults
            self.trigger_volume = {
                'x_min': 0.0, 'x_max': 20.0,
                'y_min': 0.0, 'y_max': 30.0,
                'z_min': 25.0, 'z_max': 50.0,
            }
            self.perturbation = np.array([2.0, 2.0, 2.0])
            self.stim_cooldown_frames = 100
            self.stim_duration = 20
            self.stim_intensity = 10
        
        # State tracking
        self.frame_count = 0
        self.volume_count = 0
        self.cooldown_counter = 0
        
        # Extracted state timeseries
        self.x_history = []
        self.y_history = []
        self.z_history = []
        self.frame_indices = []
        
        # Stimulus event tracking
        self.stim_events = []
        
        logger.info(
            f"LorenzDemoAlgorithm initialized with trigger volume: "
            f"x=[{self.trigger_volume['x_min']:.1f}, {self.trigger_volume['x_max']:.1f}], "
            f"y=[{self.trigger_volume['y_min']:.1f}, {self.trigger_volume['y_max']:.1f}], "
            f"z=[{self.trigger_volume['z_min']:.1f}, {self.trigger_volume['z_max']:.1f}]. "
            f"Algorithm perturbation: {self.perturbation}"
        )
    
    def initialize_model(self):
        """Initialize the algorithm model."""
        logger.info("LorenzDemoAlgorithm model initialized")
    
    def _find_local_maxima(self, image: np.ndarray, num_peaks: int = 3) -> np.ndarray:
        """
        Find local maxima in image.
        
        Args:
            image: Input image
            num_peaks: Number of peaks to find
            
        Returns:
            Array of shape (num_peaks, 2) with (row, col) coordinates
        """
        # # Apply Gaussian filter to reduce noise
        smoothed = ndimage.gaussian_filter(image.astype(float), sigma=1.0)
        
        # # Find local maxima
        local_max = ndimage.maximum_filter(smoothed, size=3)
        maxima_mask = (smoothed == local_max) & (smoothed > np.percentile(smoothed, 95))
        
        # Get coordinates of maxima
        maxima_coords = np.argwhere(maxima_mask)
        
        if len(maxima_coords) == 0:
            # No maxima found, return zeros
            return np.zeros((num_peaks, 2), dtype=int)
        
        # Get intensities at maxima locations
        maxima_intensities = smoothed[maxima_coords[:, 0], maxima_coords[:, 1]]
        
        # Sort by intensity and take top num_peaks
        sorted_indices = np.argsort(maxima_intensities)[::-1]
        top_maxima = maxima_coords[sorted_indices[:num_peaks]]
        
        # Pad with zeros if we found fewer than num_peaks
        if len(top_maxima) < num_peaks:
            padding = np.zeros((num_peaks - len(top_maxima), 2), dtype=int)
            top_maxima = np.vstack([top_maxima, padding])
        
        return top_maxima
    
    def _pixel_coords_to_lorenz_state(
        self,
        coords: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert pixel coordinates back to Lorenz state.
        
        This is the inverse of the encoding in LorenzCamera.
        
        Args:
            coords: Array of shape (3, 2) with (row, col) for each pixel
            image_shape: (height, width) of image
            
        Returns:
            Lorenz state [x, y, z]
        """
        height, width = image_shape
        pixel_radius = 2  # Should match camera setting
        
        # Sort coordinates by column (left to right) to get x, y, z order
        sorted_indices = np.argsort(coords[:, 1])
        coords_sorted = coords[sorted_indices]
        
        # Scaling factors (inverse of camera encoding)
        x_scale = (width - 2 * pixel_radius - 1) / 40.0
        x_offset = 20.0
        
        y_scale = (height - 2 * pixel_radius - 1) / 60.0
        y_offset = 30.0
        
        z_scale = (height - 2 * pixel_radius - 1) / 50.0
        
        # Extract x from leftmost pixel (vertical position)
        x_row = coords_sorted[0, 0]
        x = (x_row - pixel_radius) / x_scale - x_offset
        
        # Extract y from middle pixel (vertical position)
        y_row = coords_sorted[1, 0]
        y = (y_row - pixel_radius) / y_scale - y_offset
        
        # Extract z from rightmost pixel (vertical position)
        z_row = coords_sorted[2, 0]
        z = (z_row - pixel_radius) / z_scale
        
        return np.array([x, y, z])
    
    def process_frame(self, img: np.ndarray, zndx: int):
        """
        Process a single frame.
        
        Args:
            img: Image array
            zndx: Z-plane index
        """
        self.frame_count += 1
        
        # Only process if single plane (z-stack not expected for this demo)
        if zndx == 0:
            # Find 3 brightest local maxima
            maxima_coords = self._find_local_maxima(img, num_peaks=3)
            
            # Convert to Lorenz state
            state = self._pixel_coords_to_lorenz_state(maxima_coords, img.shape)
            
            # Store state
            self.x_history.append(state[0])
            self.y_history.append(state[1])
            self.z_history.append(state[2])
            self.frame_indices.append(self.frame_count)
            
            # Log periodically
            if self.frame_count % 50 == 0:
                logger.info(
                    f"Frame {self.frame_count}: "
                    f"Lorenz state = [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]"
                )
        
        # Update volume count
        if zndx == self.zsize - 1:
            self.volume_count += 1
    
    def process_volume(self):
        """Process completed volume (not used in this demo)."""
        pass
    
    def _is_in_trigger_volume(self, state: np.ndarray) -> bool:
        """
        Check if state is inside trigger volume.
        
        Args:
            state: Lorenz state [x, y, z]
            
        Returns:
            True if inside trigger volume
        """
        x, y, z = state
        
        in_x = self.trigger_volume['x_min'] <= x <= self.trigger_volume['x_max']
        in_y = self.trigger_volume['y_min'] <= y <= self.trigger_volume['y_max']
        in_z = self.trigger_volume['z_min'] <= z <= self.trigger_volume['z_max']
        
        return in_x and in_y and in_z
    
    def check_stim(self, image_ndx: int, cooldown_counter: int = 0) -> Tuple[Dict, int]:
        """
        Check if stimulus should be triggered based on current state.
        
        Args:
            image_ndx: Current image index
            cooldown_counter: Current cooldown value
            
        Returns:
            tuple: (stim_params dict, new_cooldown_counter)
        """
        # Use the provided cooldown counter if passed in
        if cooldown_counter > 0:
            self.cooldown_counter = cooldown_counter
        
        # Decrement cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return {}, self.cooldown_counter
        
        # Check if we have state data
        if len(self.x_history) == 0:
            return {}, 0
        
        # Get most recent state
        current_state = np.array([
            self.x_history[-1],
            self.y_history[-1],
            self.z_history[-1]
        ])
        
        # Check if in trigger volume
        if self._is_in_trigger_volume(current_state):
            logger.info(
                f"Frame {image_ndx}: Lorenz state {current_state} entered trigger volume. "
                f"Triggering stimulus."
            )
            
            # Create stimulus parameters -- isn't used in this demo instance but is common usage
            stim_params = {
                'stim_on': image_ndx + 1,
                'stim_off': image_ndx + self.stim_duration,
                'event': {
                    'stim_intensity': self.stim_intensity,
                    'perturbation': self.perturbation,
                    'lorenz_state': current_state.tolist(),
                    'frame': image_ndx,
                }
            }
            
            # Record event
            self.stim_events.append({
                'frame': image_ndx,
                'state': current_state.tolist(),
                'perturbation': self.perturbation,
            })
            
            # Set cooldown
            self.cooldown_counter = self.stim_cooldown_frames

            logging.info(f'Producing LorenzDemoAlgorithm stim_params: {stim_params}')
            
            return stim_params, self.cooldown_counter
        
        return {}, 0
    
    def get_metadata(self, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return metadata captured during runtime.
        
        Args:
            args: Optional args dict
            
        Returns:
            Dictionary containing algorithm metadata
        """
        return {
            "algorithm_type": "LorenzDemoAlgorithm",
            "frames_processed": self.frame_count,
            "volumes_processed": self.volume_count,
            
            # Trigger configuration
            "trigger_volume": self.trigger_volume,
            "perturbation": self.perturbation,
            
            # Extracted state timeseries
            "x_history": self.x_history,
            "y_history": self.y_history,
            "z_history": self.z_history,
            "frame_indices": self.frame_indices,
            
            # Stimulus events
            "stim_events": self.stim_events,
            "num_stim_events": len(self.stim_events),
            
            # Statistics
            "state_stats": {
                "x_mean": float(np.mean(self.x_history)) if self.x_history else 0,
                "y_mean": float(np.mean(self.y_history)) if self.y_history else 0,
                "z_mean": float(np.mean(self.z_history)) if self.z_history else 0,
                "x_std": float(np.std(self.x_history)) if self.x_history else 0,
                "y_std": float(np.std(self.y_history)) if self.y_history else 0,
                "z_std": float(np.std(self.z_history)) if self.z_history else 0,
            }
        }
    
    def plot_model(self, show_plot: bool = False, savefilename: Optional[str] = None):
        """
        Plot the Lorenz attractor trajectory in 3D phase space.
        
        Args:
            show_plot: Whether to display the plot
            savefilename: If provided, save plot to this file
        """
        if len(self.x_history) < 2:
            logger.warning("Not enough data to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        fig = plt.figure(figsize=(12, 10))
        
        # 3D trajectory plot
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        
        # Plot trajectory
        ax.plot(self.x_history, self.y_history, self.z_history, 
                'b-', alpha=0.6, linewidth=0.5, label='Trajectory')
        
        # Mark stimulus events
        if self.stim_events:
            stim_frames = [e['frame'] for e in self.stim_events]
            stim_indices = [self.frame_indices.index(f) for f in stim_frames 
                           if f in self.frame_indices]
            
            stim_x = [self.x_history[i] for i in stim_indices]
            stim_y = [self.y_history[i] for i in stim_indices]
            stim_z = [self.z_history[i] for i in stim_indices]
            
            ax.scatter(stim_x, stim_y, stim_z, c='red', s=100, 
                      marker='*', label='Stimulus Events', zorder=5)
        
        # Draw trigger volume
        x_min, x_max = self.trigger_volume['x_min'], self.trigger_volume['x_max']
        y_min, y_max = self.trigger_volume['y_min'], self.trigger_volume['y_max']
        z_min, z_max = self.trigger_volume['z_min'], self.trigger_volume['z_max']
        
        # Draw box edges
        edges = [
            [[x_min, x_max], [y_min, y_min], [z_min, z_min]],
            [[x_min, x_max], [y_max, y_max], [z_min, z_min]],
            [[x_min, x_max], [y_min, y_min], [z_max, z_max]],
            [[x_min, x_max], [y_max, y_max], [z_max, z_max]],
            [[x_min, x_min], [y_min, y_max], [z_min, z_min]],
            [[x_max, x_max], [y_min, y_max], [z_min, z_min]],
            [[x_min, x_min], [y_min, y_max], [z_max, z_max]],
            [[x_max, x_max], [y_min, y_max], [z_max, z_max]],
            [[x_min, x_min], [y_min, y_min], [z_min, z_max]],
            [[x_max, x_max], [y_min, y_min], [z_min, z_max]],
            [[x_min, x_min], [y_max, y_max], [z_min, z_max]],
            [[x_max, x_max], [y_max, y_max], [z_min, z_max]],
        ]
        
        for edge in edges:
            ax.plot(edge[0], edge[1], edge[2], 'g--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz Attractor - 3D Phase Space')
        ax.legend()
        
        # Time series plots
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(self.frame_indices, self.x_history, 'r-', label='X', alpha=0.7)
        ax2.plot(self.frame_indices, self.y_history, 'g-', label='Y', alpha=0.7)
        ax2.plot(self.frame_indices, self.z_history, 'b-', label='Z', alpha=0.7)
        
        # Mark stimuli
        if self.stim_events:
            for event in self.stim_events:
                ax2.axvline(event['frame'], color='red', alpha=0.3, linestyle='--')
        
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('State Value')
        ax2.set_title('Lorenz State Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # XY projection
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(self.x_history, self.y_history, 'b-', alpha=0.6, linewidth=0.5)
        if self.stim_events:
            ax3.scatter(stim_x, stim_y, c='red', s=100, marker='*', zorder=5)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('XY Projection')
        ax3.grid(True, alpha=0.3)
        
        # XZ projection
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(self.x_history, self.z_history, 'b-', alpha=0.6, linewidth=0.5)
        if self.stim_events:
            ax4.scatter(stim_x, stim_z, c='red', s=100, marker='*', zorder=5)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Z')
        ax4.set_title('XZ Projection')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if savefilename:
            plt.savefig(savefilename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {savefilename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def close(self):
        """Clean up resources."""
        logger.info(
            f"LorenzDemoAlgorithm closing. Processed {self.frame_count} frames, "
            f"triggered {len(self.stim_events)} stimuli"
        )
