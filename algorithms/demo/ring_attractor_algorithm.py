"""
Ring Attractor Demo Algorithm

Extracts puncta position from images and tracks state trajectory on dual ring attractors.
Provides real-time visualization with interactive stimulus control using -30 to +30 perturbation
and omega perturbation from -5 to +5.

The algorithm:
- Finds puncta via centroid of brightest pixels
- Stores raw XY positions from image
- Provides interactive GUI with manual stimulus triggering
- Displays raw XY trajectory in Cartesian state space with ring overlays
- Slider controls radial perturbation from -30 (inward) to +30 (outward)
- Slider controls omega perturbation from -5 (slower) to +5 (faster)
"""

import logging
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import ndimage

logger = logging.getLogger(__name__)

# Import PyQt and pyqtgraph
try:
    from pyqtgraph.Qt import QtCore, QtWidgets
    import pyqtgraph as pg
except ImportError:
    logger.error("PyQt or pyqtgraph not available")
    raise

# Stylization for app
from pathlib import Path
ROOTDIR = Path(__file__).resolve().parents[2]
CSS_PATH = ROOTDIR  / "style" / "css" / "Ubuntu.qss"

class RingAttractorAlgorithm:
    """
    Algorithm for ring attractor closed-loop demo.
    
    Extracts puncta position, monitors trajectory, and provides
    manual stimulus control with visualization.
    """
    
    def __init__(
        self,
        algorithm_config: Optional['AlgorithmConfig'] = None,
        experiment_config: Optional['ExperimentConfig'] = None,
        hardware_manager: Optional['HardwareManager'] = None,
        local_handles: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None
    ):
        """Initialize ring attractor algorithm."""
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
        self.samples_to_grab = experiment_config.acquisition.num_samples
        self.saveroot = experiment_config.output_dir
        
        # Get algorithm-specific parameters
        if algorithm_config:
            params = algorithm_config.algorithm_params
            
            # Ring parameters (from hardware config)
            hw_config = hardware_manager.config
            ring_params = hw_config.ring_params
            self.inner_radius = ring_params.get('inner_radius', 3.0)
            self.outer_radius = ring_params.get('outer_radius', 6.0)
            self.image_width = ring_params.get('image_width', 100)
            self.image_height = ring_params.get('image_height', 100)
            
            # Stimulus parameters
            self.stim_cooldown_frames = params.stim_cooldown_frames
            self.default_stim_intensity = 0  # Default to 0 (no perturbation)
            self.default_omega_perturbation = 0  # Default to 0 (no omega change)
            self.visualize_real_time = params.visualize_real_time
            
            # Auto-stim parameters
            self.auto_stim_enabled = params.auto_stim_enabled
            self.auto_stim_theta_min = params.auto_stim_theta_min
            self.auto_stim_theta_max = params.auto_stim_theta_max

            # Trajectory visualization
            self.fading_trajectory_samples = params.fading_trajectory_samples

            # Screenshot frequency
            try:
                self.gui_screenshot_freq = algorithm_config.gui_params.gui_screenshot_freq
            except Exception:
                self.gui_screenshot_freq = 0
        else:
            # Defaults
            self.inner_radius = 3.0
            self.outer_radius = 6.0
            self.image_width = 100
            self.image_height = 100
            self.stim_cooldown_frames = 50
            self.default_stim_intensity = 0
            self.default_omega_perturbation = 0
            self.visualize_real_time = False
            self.auto_stim_enabled = False
            self.auto_stim_theta_min = 0.0
            self.auto_stim_theta_max = np.pi / 4
            self.fading_trajectory_samples = 100
            self.gui_screenshot_freq = 0
        
        # Image center (for converting to centered coordinates)
        self.center_x = self.image_width / 2.0
        self.center_y = self.image_height / 2.0
        
        # State tracking
        self.frame_count = 0
        self.cooldown_counter = 0
        self.transition_count = 0
        
        # Extracted state timeseries - RAW XY positions
        self.x_history = []  # Raw X pixel positions (centered)
        self.y_history = []  # Raw Y pixel positions (centered)
        self.theta_history = []  # Derived theta for auto-stim
        self.ring_history = []  # Derived ring classification
        self.frame_indices = [] 
        self.stim_events = []
        
        # Manual stimulus control
        self.manual_stim_pending = False
        self.current_stim_intensity = self.default_stim_intensity
        self.current_omega_perturbation = self.default_omega_perturbation
        
        # Initialize visualizer
        self.visualizer = None
        if self.visualize_real_time:
            try:
                self.visualizer = RingVisualizer(self)
            except Exception as e:
                logger.warning(f"Could not initialize visualizer: {e}")
        
        logger.info(
            f"RingAttractorAlgorithm initialized with rings at r={self.inner_radius}, {self.outer_radius}"
        )
    
    def initialize_model(self):
        """Initialize the algorithm model."""
        logger.info("RingAttractorAlgorithm model initialized")
    
    def _find_puncta_centroid(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Find puncta position via centroid of brightest pixels.
        
        Args:
            image: Input image
            
        Returns:
            (x, y) centroid position in pixel coordinates
        """
        # Threshold at high percentile to isolate puncta
        threshold = np.percentile(image, 99)
        mask = image > threshold
        
        if not np.any(mask):
            # No bright pixels found, return center
            return self.image_width / 2.0, self.image_height / 2.0
        
        # Compute centroid
        coords = np.argwhere(mask)
        centroid_y = np.mean(coords[:, 0])
        centroid_x = np.mean(coords[:, 1])
        
        return centroid_x, centroid_y
    
    def _pixel_to_polar(
        self, 
        x: float, 
        y: float
    ) -> Tuple[float, float, int]:
        """
        Convert pixel coordinates to polar (theta, r) and classify ring.
        
        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            
        Returns:
            Tuple of (theta, r, ring_index)
        """
        # Relative to center
        dx = x - self.center_x
        dy = y - self.center_y
        
        # Polar coordinates
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        # Wrap theta to [0, 2π)
        if theta < 0:
            theta += 2 * np.pi
        
        # Classify ring (nearest)
        dist_inner = abs(r - self.inner_radius)
        dist_outer = abs(r - self.outer_radius)
        ring_index = 0 if dist_inner < dist_outer else 1
        
        return theta, r, ring_index
    
    def process_sample(self, img: np.ndarray, sample_ndx: int):
        """
        Process a single frame.
        
        Args:
            img: Image array
            sample_ndx: Sample index
        """
        self.frame_count += 1
        
        # Find puncta centroid (raw pixel coordinates)
        x_pixel, y_pixel = self._find_puncta_centroid(img)
        
        # Convert to centered coordinates for state space
        x_centered = x_pixel - self.center_x
        y_centered = y_pixel - self.center_y
        
        # Convert to polar for ring classification and auto-stim
        theta, r, ring_idx = self._pixel_to_polar(x_pixel, y_pixel)
        
        # Store RAW centered XY positions
        self.x_history.append(x_centered)
        self.y_history.append(y_centered)
        self.theta_history.append(theta)
        self.ring_history.append(ring_idx)
        self.frame_indices.append(self.frame_count)
        
        # Check for ring transitions
        if len(self.ring_history) > 1:
            if self.ring_history[-1] != self.ring_history[-2]:
                self.transition_count += 1
                logger.info(f"Frame {self.frame_count}: Ring transition detected (total: {self.transition_count})")
        
        # Update visualizer
        if self.visualizer:
            self.visualizer.update_image(img, x_pixel, y_pixel)
            self.visualizer.update_trajectory()
            self.visualizer.update_info_text()
            self.visualizer.process_events()

        self._maybe_save_screenshot(sample_ndx)

        # Log periodically
        if self.frame_count % 100 == 0:
            logger.info(
                f"Frame {self.frame_count}: x={x_centered:.1f}, y={y_centered:.1f}, r={r:.1f}, ring={ring_idx}"
            )
    
    def _maybe_save_screenshot(self, sample_ndx: int):
        """Save a screenshot of the GUI window at the configured frequency."""
        if not self.gui_screenshot_freq or sample_ndx % self.gui_screenshot_freq != 0:
            return
        if not self.visualizer:
            return
        try:
            screenshot_dir = Path(self.saveroot) / "gui_screenshot"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = screenshot_dir / f"screenshot_{sample_ndx:06d}.png"
            pixmap = self.visualizer.window.grab()
            pixmap.save(str(path), "PNG")
            logger.debug(f"Saved GUI screenshot: {path}")
        except Exception as err:
            logger.warning(f"Failed to save GUI screenshot at sample {sample_ndx}: {err}")

    def process_volume(self):
        """Process completed volume (not used in this demo)."""
        pass
    
    def check_stim(self, image_ndx: int, cooldown_counter: int = 0) -> Tuple[Dict, int]:
        """
        Check if stimulus should be triggered.
        
        Args:
            image_ndx: Current image index
            cooldown_counter: Current cooldown value
            
        Returns:
            tuple: (stim_params dict, new_cooldown_counter)
        """
        # Use provided cooldown counter
        if cooldown_counter > 0:
            self.cooldown_counter = cooldown_counter
        
        # Decrement cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return {}, self.cooldown_counter
        
        # Check manual trigger
        if self.manual_stim_pending:
            self.manual_stim_pending = False
            
            stim_params = {
                'stim_on': image_ndx + 1,
                'stim_off': image_ndx + 20,  # arbitrary duration
                'event': {
                    'stim_intensity': self.current_stim_intensity,
                    'omega_perturbation': self.current_omega_perturbation,
                    'frame': image_ndx,
                    'trigger_type': 'manual'
                }
            }
            
            self.stim_events.append({
                'frame': image_ndx,
                'x': self.x_history[-1] if self.x_history else 0,
                'y': self.y_history[-1] if self.y_history else 0,
                'theta': self.theta_history[-1] if self.theta_history else 0,
                'ring': self.ring_history[-1] if self.ring_history else 0,
                'stim_intensity': self.current_stim_intensity,
                'omega_perturbation': self.current_omega_perturbation,
                'trigger_type': 'manual'
            })
            
            self.cooldown_counter = self.stim_cooldown_frames
            logger.info(
                f"Manual stimulus triggered at frame {image_ndx}, "
                f"perturbation={self.current_stim_intensity}, "
                f"omega_perturbation={self.current_omega_perturbation}"
            )
            
            return stim_params, self.cooldown_counter
        
        # Check auto-trigger
        if self.visualizer.auto_stim_checkbox.isChecked() and len(self.theta_history) > 0:
            theta = self.theta_history[-1]
            
            if self.auto_stim_theta_min <= theta <= self.auto_stim_theta_max:

                intensity = self.visualizer.intensity_slider.value()
                omega_perturbation = self.visualizer.omega_slider.value() / 10.0  # Convert to -10.0 to +10.0
                stim_params = {
                    'stim_on': image_ndx + 1,
                    'stim_off': image_ndx + 20,
                    'event': {
                        'stim_intensity': intensity,
                        'omega_perturbation': omega_perturbation,  # No omega perturbation for auto-stim
                        'frame': image_ndx,
                        'trigger_type': 'auto',
                        'theta': theta
                    }
                }
                
                self.stim_events.append({
                    'frame': image_ndx,
                    'x': self.x_history[-1],
                    'y': self.y_history[-1],
                    'theta': theta,
                    'ring': self.ring_history[-1],
                    'stim_intensity': intensity,
                    'omega_perturbation': 0,
                    'trigger_type': 'auto'
                })
                
                self.cooldown_counter = self.stim_cooldown_frames
                logger.info(f"Auto stimulus triggered at frame {image_ndx}, theta={theta:.3f}")
                
                return stim_params, self.cooldown_counter
        
        return {}, 0
    
    def trigger_manual_stimulus(self, intensity: float, omega_perturbation: float):
        """
        Trigger manual stimulus from GUI.
        
        Args:
            intensity: Perturbation strength (-30 to +30)
                Positive = push outward
                Negative = pull inward
            omega_perturbation: Angular velocity perturbation (-5 to +5)
                Positive = speed up rotation
                Negative = slow down rotation
        """
        self.current_stim_intensity = intensity
        self.current_omega_perturbation = omega_perturbation
        self.manual_stim_pending = True
        logger.info(
            f"Manual stimulus queued with perturbation {intensity}, "
            f"omega_perturbation {omega_perturbation}"
        )
    
    def set_auto_stim_enabled(self, enabled: bool):
        """Set auto-stim enabled state."""
        self.auto_stim_enabled = enabled
        logger.info(f"Auto-stim {'enabled' if enabled else 'disabled'}")
    
    def get_metadata(self, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return metadata captured during runtime."""
        return {
            "algorithm_type": "RingAttractorAlgorithm",
            "frames_processed": self.frame_count,
            
            # Ring configuration
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            
            # Extracted state timeseries
            "x_history": self.x_history,
            "y_history": self.y_history,
            "theta_history": self.theta_history,
            "ring_history": self.ring_history,
            "frame_indices": self.frame_indices,
            
            # Stimulus events
            "stim_events": self.stim_events,
            "num_stim_events": len(self.stim_events),
            "num_transitions": self.transition_count,
            
            # Statistics
            "state_stats": {
                "mean_x": float(np.mean(self.x_history)) if self.x_history else 0,
                "std_x": float(np.std(self.x_history)) if self.x_history else 0,
                "mean_y": float(np.mean(self.y_history)) if self.y_history else 0,
                "std_y": float(np.std(self.y_history)) if self.y_history else 0,
                "mean_theta": float(np.mean(self.theta_history)) if self.theta_history else 0,
                "std_theta": float(np.std(self.theta_history)) if self.theta_history else 0,
                "time_inner_ring": sum(1 for r in self.ring_history if r == 0) / len(self.ring_history) if self.ring_history else 0,
                "time_outer_ring": sum(1 for r in self.ring_history if r == 1) / len(self.ring_history) if self.ring_history else 0,
            }
        }
    
    def plot_model(self, show_plot: bool = False, savefilename: Optional[str] = None):
        """Plot the ring attractor trajectory."""
        if len(self.x_history) < 2:
            logger.warning("Not enough data to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # XY state space trajectory
        ax = axes[0, 0]
        
        # Color by time
        colors = np.arange(len(self.x_history))
        scatter = ax.scatter(self.x_history, self.y_history, c=colors, s=1, cmap='viridis', alpha=0.5)
        
        # Draw ring circles
        circle_inner = plt.Circle((0, 0), self.inner_radius, fill=False, color='blue', linestyle='--', alpha=0.5)
        circle_outer = plt.Circle((0, 0), self.outer_radius, fill=False, color='red', linestyle='--', alpha=0.5)
        ax.add_patch(circle_inner)
        ax.add_patch(circle_outer)
        
        # Mark stimuli
        if self.stim_events:
            stim_x = [e['x'] for e in self.stim_events]
            stim_y = [e['y'] for e in self.stim_events]
            ax.scatter(stim_x, stim_y, c='gold', s=100, marker='*', zorder=5, label='Stimulus')
        
        ax.set_xlabel('X (pixels, centered)')
        ax.set_ylabel('Y (pixels, centered)')
        ax.set_title('XY State Space Trajectory')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Time (frame)')
        
        # X position time series
        ax = axes[0, 1]
        ax.plot(self.frame_indices, self.x_history, 'r-', alpha=0.7, linewidth=0.5, label='X')
        for event in self.stim_events:
            ax.axvline(event['frame'], color='gold', alpha=0.3, linestyle='--')
        ax.set_xlabel('Frame')
        ax.set_ylabel('X Position (pixels)')
        ax.set_title('X Position Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Y position time series
        ax = axes[1, 0]
        ax.plot(self.frame_indices, self.y_history, 'b-', alpha=0.7, linewidth=0.5, label='Y')
        for event in self.stim_events:
            ax.axvline(event['frame'], color='gold', alpha=0.3, linestyle='--')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Y Position Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ring index time series
        ax = axes[1, 1]
        ax.plot(self.frame_indices, self.ring_history, 'k-', linewidth=0.5)
        for event in self.stim_events:
            ax.axvline(event['frame'], color='gold', alpha=0.3, linestyle='--')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Ring Index')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Inner', 'Outer'])
        ax.set_title(f'Ring Transitions (Total: {self.transition_count})')
        ax.grid(True, alpha=0.3)
        
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
        if self.visualizer:
            self.visualizer.close()
        logger.info(
            f"RingAttractorAlgorithm closing. Processed {self.frame_count} frames, "
            f"{self.transition_count} transitions, {len(self.stim_events)} stimuli"
        )


class RingVisualizer:
    """Real-time visualization with XY state space plot showing raw positions."""
    
    def __init__(self, algorithm: 'RingAttractorAlgorithm'):
        """Initialize visualizer with XY state space plot."""
        self.algorithm = algorithm
        
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.pg = pg
        
        # Create Qt application
        self.app = pg.mkQApp("RingVisualizer")
        with open(CSS_PATH, 'r') as f:
            self.app.setStyleSheet(f.read())
        
        # Create main window
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Ring Attractor - XY State Space Demo")
        self.window.resize(1200, 700)
        
        # Create layout - 2 columns
        self.layout = QtWidgets.QGridLayout()
        self.window.setLayout(self.layout)
        
        # === LEFT: Image display ===
        self.image_widget = pg.ImageView()
        self.image_widget.ui.roiBtn.hide()
        self.image_widget.ui.menuBtn.hide()
        self.layout.addWidget(self.image_widget, 0, 0, 2, 1)
        
        # === RIGHT: XY state space ===
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setTitle("XY State Space (Raw Positions)")
        self.plot_widget.setLabel('left', 'Y (pixels, centered)')
        self.plot_widget.setLabel('bottom', 'X (pixels, centered)')
        self.layout.addWidget(self.plot_widget, 0, 1, 1, 1)
        
        # Draw ring circles
        self._draw_ring_circles()
        
        # Trajectory plot
        self.trajectory_plot = self.plot_widget.plot(pen=pg.mkPen('c', width=2))
        self.current_pos_plot = self.plot_widget.plot(
            pen=None, symbol='o', symbolSize=10, symbolBrush='y'
        )
        
        # === BOTTOM RIGHT: Control panel ===
        self.control_widget = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QVBoxLayout()
        self.control_widget.setLayout(self.control_layout)
        self.layout.addWidget(self.control_widget, 1, 1, 1, 1)
        
        # Trigger button
        self.trigger_button = QtWidgets.QPushButton("Apply Perturbation")
        self.trigger_button.clicked.connect(self._on_trigger_clicked)
        self.control_layout.addWidget(self.trigger_button)
        
        # Radial perturbation slider (-30 to +30)
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("Radial:"))
        self.intensity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.intensity_slider.setMinimum(-100)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(0)
        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        slider_layout.addWidget(self.intensity_slider)
        self.intensity_label = QtWidgets.QLabel("0 (none)")
        self.intensity_label.setMinimumWidth(100)
        slider_layout.addWidget(self.intensity_label)
        self.control_layout.addLayout(slider_layout)
        
        # Omega perturbation slider (-5 to +5)
        omega_slider_layout = QtWidgets.QHBoxLayout()
        omega_slider_layout.addWidget(QtWidgets.QLabel("Omega:"))
        self.omega_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.omega_slider.setMinimum(-100)  # -10.0 with 0.1 resolution
        self.omega_slider.setMaximum(100)   # +10.0 with 0.1 resolution
        self.omega_slider.setValue(0)
        self.omega_slider.valueChanged.connect(self._on_omega_changed)
        omega_slider_layout.addWidget(self.omega_slider)
        self.omega_label = QtWidgets.QLabel("0.0 (none)")
        self.omega_label.setMinimumWidth(100)
        omega_slider_layout.addWidget(self.omega_label)
        self.control_layout.addLayout(omega_slider_layout)
        
        # Helper text
        helper_text = QtWidgets.QLabel(
            "<small><b>Radial Perturbation:</b><br>"
            "• Positive (+) = push outward<br>"
            "• Negative (-) = pull inward<br>"
            "• ~±15 will switch rings<br><br>"
            "<b>Omega Perturbation:</b><br>"
            "• Positive (+) = speed up rotation<br>"
            "• Negative (-) = slow down rotation</small>"
        )
        helper_text.setStyleSheet("QLabel { color: #666; padding: 5px; }")
        self.control_layout.addWidget(helper_text)
        
        # Auto-stim checkbox
        self.auto_stim_checkbox = QtWidgets.QCheckBox("Auto-trigger at θ ∈ [0, π/4]")
        self.auto_stim_checkbox.setChecked(self.algorithm.auto_stim_enabled)
        self.auto_stim_checkbox.stateChanged.connect(self._on_auto_stim_changed)
        self.control_layout.addWidget(self.auto_stim_checkbox)
        
        # Info text
        self.info_text = QtWidgets.QLabel()
        self.info_text.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.info_text.setStyleSheet("QLabel { background-color: white; padding: 10px; }")
        self.control_layout.addWidget(self.info_text)
        
        self.window.show()
        logger.info("RingVisualizer with XY state space initialized")
    
    def _draw_ring_circles(self):
        """Draw dotted circles for ring attractors."""
        # Inner ring
        theta = np.linspace(0, 2*np.pi, 100)
        x_inner = self.algorithm.inner_radius * np.cos(theta)
        y_inner = self.algorithm.inner_radius * np.sin(theta)
        self.plot_widget.plot(x_inner, y_inner, pen=pg.mkPen('b', width=2, style=QtCore.Qt.DotLine))
        
        # Outer ring
        x_outer = self.algorithm.outer_radius * np.cos(theta)
        y_outer = self.algorithm.outer_radius * np.sin(theta)
        self.plot_widget.plot(x_outer, y_outer, pen=pg.mkPen('r', width=2, style=QtCore.Qt.DotLine))
    
    def _on_trigger_clicked(self):
        """Handle manual trigger button click."""
        intensity = self.intensity_slider.value()
        omega_perturbation = self.omega_slider.value() / 10.0  # Convert to -10.0 to +10.0
        self.algorithm.trigger_manual_stimulus(intensity, omega_perturbation)
    
    def _on_intensity_changed(self, value):
        """Handle intensity slider change."""
        if value > 0:
            label = f"+{value} (outward)"
        elif value < 0:
            label = f"{value} (inward)"
        else:
            label = "0 (none)"
        self.intensity_label.setText(label)
        self.algorithm.current_stim_intensity = value
    
    def _on_omega_changed(self, value):
        """Handle omega slider change."""
        omega_value = value / 10.0  # Convert to -5.0 to +5.0
        if omega_value > 0:
            label = f"+{omega_value:.1f} (faster)"
        elif omega_value < 0:
            label = f"{omega_value:.1f} (slower)"
        else:
            label = "0.0 (none)"
        self.omega_label.setText(label)
        self.algorithm.current_omega_perturbation = omega_value
    
    def _on_auto_stim_changed(self, state):
        """Handle auto-stim checkbox change."""
        enabled = state == QtCore.Qt.Checked
        self.algorithm.set_auto_stim_enabled(enabled)
    
    def update_image(self, img: np.ndarray, puncta_x: float, puncta_y: float):
        """Update image display with puncta marker."""
        self.image_widget.setImage(img.T, autoLevels=False, autoRange=False)
    
    def update_trajectory(self):
        """Update XY state space trajectory plot."""
        if len(self.algorithm.x_history) < 2:
            return
        
        # Get last N points for fading trail
        N = min(self.algorithm.fading_trajectory_samples, len(self.algorithm.x_history))
        x_coords = self.algorithm.x_history[-N:]
        y_coords = self.algorithm.y_history[-N:]
        
        # Plot trajectory
        self.trajectory_plot.setData(x_coords, y_coords)
        
        # Current position
        if x_coords:
            self.current_pos_plot.setData([x_coords[-1]], [y_coords[-1]])
    
    def update_info_text(self):
        """Update info text."""
        if not self.algorithm.x_history:
            return
            
        x = self.algorithm.x_history[-1]
        y = self.algorithm.y_history[-1]
        theta = self.algorithm.theta_history[-1] if self.algorithm.theta_history else 0
        ring_idx = self.algorithm.ring_history[-1] if self.algorithm.ring_history else 0
        r = np.sqrt(x**2 + y**2)
        
        info = f"""
        <b>Ring Attractor Closed-Loop Demo</b><br><br>
        <b>Frame:</b> {self.algorithm.frame_count} / {self.algorithm.samples_to_grab}<br>
        <b>Current State:</b><br>
        &nbsp;&nbsp;X = {x:.1f} px<br>
        &nbsp;&nbsp;Y = {y:.1f} px<br>
        &nbsp;&nbsp;R = {r:.1f} px<br>
        &nbsp;&nbsp;Theta = {theta:.3f} rad ({np.degrees(theta):.1f}°)<br>
        &nbsp;&nbsp;Ring = {'Inner' if ring_idx == 0 else 'Outer'}<br>
        <br>
        <b>Stimulus Events:</b> {len(self.algorithm.stim_events)}<br>
        <b>Ring Transitions:</b> {self.algorithm.transition_count}<br>
        <b>Cooldown:</b> {self.algorithm.cooldown_counter} frames<br>
        """
        self.info_text.setText(info)
    
    def process_events(self):
        """Process Qt events."""
        self.app.processEvents()
    
    def close(self):
        """Close visualizer."""
        self.window.close()
        logger.info("RingVisualizer closed")
