"""
Display RGB Algorithm - Real-time visualization of RGB screen capture.

Simply displays incoming RGB frames with pyqtgraph for live monitoring.
Can be extended to trigger input events based on visual analysis.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class DisplayRGBAlgorithm:
    """
    Algorithm for displaying RGB screen capture data in real-time.
    
    Provides live visualization and optional trigger logic for input events.
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
        Initialize display RGB algorithm.
        
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
        self.frames_to_grab = gooey_args.get("total_frames", 100)
        
        # Algorithm parameters
        if algorithm_config:
            params = algorithm_config.algorithm_params
            self.visualize_real_time = params.visualize_real_time
            self.trigger_enabled = params.trigger_enabled
            self.trigger_color_threshold = params.trigger_color_threshold
            self.stim_cooldown_frames = params.stim_cooldown_frames
            
            # Stimulus parameters
            stim_params = algorithm_config.stimulus_params
            self.stim_duration = stim_params.duration_frames
            self.stim_keys = stim_params.keyboard_keys
            self.stim_mouse_action = stim_params.mouse_action
        else:
            # Defaults
            self.visualize_real_time = True
            self.trigger_enabled = False
            self.trigger_color_threshold = 200
            self.stim_cooldown_frames = 30
            self.stim_duration = 5
            self.stim_keys = ["space"]
            self.stim_mouse_action = "click"
        
        # State tracking
        self.frame_count = 0
        self.cooldown_counter = 0
        self.stim_events = []
        
        # Color statistics history
        self.mean_brightness_history = []
        self.mean_red_history = []
        self.mean_green_history = []
        self.mean_blue_history = []
        
        # Initialize visualizer
        self.visualizer = None
        if self.visualize_real_time:
            try:
                self.visualizer = RGBVisualizer(self)
            except Exception as e:
                logger.warning(f"Could not initialize visualizer: {e}")
        
        logger.info(
            f"DisplayRGBAlgorithm initialized: "
            f"visualization={self.visualize_real_time}, "
            f"trigger_enabled={self.trigger_enabled}"
        )
    
    def initialize_model(self):
        """Initialize the algorithm model."""
        logger.info("DisplayRGBAlgorithm model initialized")
    
    def process_sample(self, img: np.ndarray, sample_ndx: int):
        """
        Process a single RGB frame.
        
        Args:
            img: RGB image array (H, W, 3)
            sample_ndx: Sample index
        """
        self.frame_count += 1
        
        # Calculate color statistics
        mean_brightness = np.mean(img)
        mean_red = np.mean(img[:, :, 0])
        mean_green = np.mean(img[:, :, 1])
        mean_blue = np.mean(img[:, :, 2])
        
        # Store statistics
        self.mean_brightness_history.append(mean_brightness)
        self.mean_red_history.append(mean_red)
        self.mean_green_history.append(mean_green)
        self.mean_blue_history.append(mean_blue)
        
        # Update visualizer
        if self.visualizer:
            self.visualizer.update_image(img)
            self.visualizer.update_statistics()
            self.visualizer.update_info_text()
            self.visualizer.process_events()
        
        # Log periodically
        if self.frame_count % 50 == 0:
            logger.info(
                f"Frame {self.frame_count}: "
                f"brightness={mean_brightness:.1f}, "
                f"R={mean_red:.1f}, G={mean_green:.1f}, B={mean_blue:.1f}"
            )
    
    def process_volume(self):
        """Process completed volume (not used for RGB capture)."""
        pass
    
    def _check_color_trigger(self, img: np.ndarray) -> bool:
        """
        Check if image triggers color threshold.
        
        Args:
            img: RGB image array
            
        Returns:
            True if bright pixels exceed threshold
        """
        if not self.trigger_enabled:
            return False
        
        # Count pixels above brightness threshold
        brightness = np.mean(img, axis=2)
        bright_pixels = np.sum(brightness > self.trigger_color_threshold)
        bright_fraction = bright_pixels / (img.shape[0] * img.shape[1])
        
        # Trigger if >10% of pixels are bright
        return bright_fraction > 0.1
    
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
        
        # Check for trigger (example: based on brightness)
        if len(self.mean_brightness_history) == 0:
            return {}, 0
        
        current_brightness = self.mean_brightness_history[-1]
        
        # Simple trigger: if brightness exceeds threshold
        if self.trigger_enabled and current_brightness > self.trigger_color_threshold:
            logger.info(
                f"Frame {image_ndx}: Brightness {current_brightness:.1f} "
                f"exceeded threshold {self.trigger_color_threshold}. Triggering input."
            )
            
            # Create stimulus parameters
            stim_params = {
                'stim_on': image_ndx + 1,
                'stim_off': image_ndx + self.stim_duration,
                'keys': self.stim_keys,
                'mouse_action': self.stim_mouse_action,
                'event': {
                    'brightness': current_brightness,
                    'frame': image_ndx,
                }
            }
            
            # Record event
            self.stim_events.append({
                'frame': image_ndx,
                'brightness': current_brightness,
            })
            
            # Set cooldown
            self.cooldown_counter = self.stim_cooldown_frames
            
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
            "algorithm_type": "DisplayRGBAlgorithm",
            "frames_processed": self.frame_count,
            
            # Color statistics
            "mean_brightness_history": self.mean_brightness_history,
            "mean_red_history": self.mean_red_history,
            "mean_green_history": self.mean_green_history,
            "mean_blue_history": self.mean_blue_history,
            
            # Statistics
            "overall_mean_brightness": float(np.mean(self.mean_brightness_history)) if self.mean_brightness_history else 0,
            "overall_std_brightness": float(np.std(self.mean_brightness_history)) if self.mean_brightness_history else 0,
            
            # Trigger events
            "stim_events": self.stim_events,
            "num_stim_events": len(self.stim_events),
            "trigger_enabled": self.trigger_enabled,
        }
    
    def plot_model(self, show_plot: bool = False, savefilename: Optional[str] = None):
        """
        Plot color statistics over time.
        
        Args:
            show_plot: Whether to display the plot
            savefilename: If provided, save plot to this file
        """
        if len(self.mean_brightness_history) < 2:
            logger.warning("Not enough data to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Color channels plot
        ax1 = axes[0]
        frames = list(range(len(self.mean_red_history)))
        ax1.plot(frames, self.mean_red_history, 'r-', label='Red', alpha=0.7)
        ax1.plot(frames, self.mean_green_history, 'g-', label='Green', alpha=0.7)
        ax1.plot(frames, self.mean_blue_history, 'b-', label='Blue', alpha=0.7)
        
        # Mark stimuli
        if self.stim_events:
            for event in self.stim_events:
                ax1.axvline(event['frame'], color='black', alpha=0.3, linestyle='--')
        
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Mean Channel Value')
        ax1.set_title('RGB Channel Statistics Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Brightness plot
        ax2 = axes[1]
        ax2.plot(frames, self.mean_brightness_history, 'k-', linewidth=2, label='Brightness')
        
        if self.trigger_enabled:
            ax2.axhline(self.trigger_color_threshold, color='red', 
                       linestyle='--', label='Trigger Threshold')
        
        # Mark stimuli
        if self.stim_events:
            for event in self.stim_events:
                ax2.axvline(event['frame'], color='red', alpha=0.3, linestyle='--')
        
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Mean Brightness')
        ax2.set_title('Overall Brightness Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
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
            f"DisplayRGBAlgorithm closing. Processed {self.frame_count} frames, "
            f"triggered {len(self.stim_events)} stimuli"
        )


class RGBVisualizer:
    """Real-time visualization for RGB screen capture."""
    
    def __init__(self, algorithm: 'DisplayRGBAlgorithm'):
        """Initialize visualizer."""
        self.algorithm = algorithm
        
        # Import PyQt and pyqtgraph
        try:
            from pyqtgraph.Qt import QtCore, QtWidgets
            import pyqtgraph as pg
        except ImportError:
            logger.error("PyQt or pyqtgraph not available")
            raise

        # Useful vars
        self.FIRST_IMAGE = True
        self.downsample_factor = 4

        # Qt setup
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.pg = pg
        
        # Create Qt application
        self.app = pg.mkQApp("RGBVisualizer")
        
        # Create main window
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("RGB Screen Capture - Live Display")
        self.window.resize(1200, 800)
        
        # Create layout
        self.layout = QtWidgets.QGridLayout()
        self.window.setLayout(self.layout)
        
        # Image display (top)
        self.image_widget = pg.ImageView()
        self.image_widget.ui.roiBtn.hide()
        self.image_widget.ui.menuBtn.hide()
        self.layout.addWidget(self.image_widget, 0, 0, 1, 2)
        
        # Statistics plot (bottom left)
        self.stats_widget = pg.PlotWidget()
        self.stats_widget.setLabel('left', 'Mean Value')
        self.stats_widget.setLabel('bottom', 'Frame')
        self.stats_widget.addLegend()
        self.stats_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.red_curve = self.stats_widget.plot(pen='r', name='Red')
        self.green_curve = self.stats_widget.plot(pen='g', name='Green')
        self.blue_curve = self.stats_widget.plot(pen='b', name='Blue')
        self.brightness_curve = self.stats_widget.plot(pen=(200, 200, 200), name='Brightness', width=2)
        
        self.layout.addWidget(self.stats_widget, 1, 0, 1, 1)
        
        # Info text (bottom right)
        self.info_text = QtWidgets.QLabel()
        self.info_text.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.info_text.setStyleSheet("QLabel { background-color: white; padding: 10px; }")
        self.layout.addWidget(self.info_text, 1, 1, 1, 1)
        
        self.window.show()
        
        logger.info("RGBVisualizer initialized")
    
    def update_image(self, img: np.ndarray):
        """Update image display."""
        # pyqtgraph ImageView expects (width, height, 3) with origin at bottom-left
        # Our images are (height, width, 3) so we need to transpose and flip
        if self.FIRST_IMAGE:
            self.image_widget.setImage(img.transpose(1, 0, 2), autoLevels=True, autoRange=True)
            self.FIRST_IMAGE = False
        else:
            self.image_widget.setImage(img.transpose(1, 0, 2), autoLevels=False, autoRange=False)
    
    def update_statistics(self):
        """Update statistics plot."""
        if len(self.algorithm.mean_red_history) < 2:
            return
        
        frames = list(range(len(self.algorithm.mean_red_history)))
        
        self.red_curve.setData(frames, self.algorithm.mean_red_history)
        self.green_curve.setData(frames, self.algorithm.mean_green_history)
        self.blue_curve.setData(frames, self.algorithm.mean_blue_history)
        self.brightness_curve.setData(frames, self.algorithm.mean_brightness_history)
    
    def update_info_text(self):
        """Update info text."""
        current_brightness = (
            self.algorithm.mean_brightness_history[-1]
            if self.algorithm.mean_brightness_history else 0
        )
        current_red = (
            self.algorithm.mean_red_history[-1]
            if self.algorithm.mean_red_history else 0
        )
        current_green = (
            self.algorithm.mean_green_history[-1]
            if self.algorithm.mean_green_history else 0
        )
        current_blue = (
            self.algorithm.mean_blue_history[-1]
            if self.algorithm.mean_blue_history else 0
        )
        
        info = f"""
        <b>RGB Screen Capture Display</b><br><br>
        <b>Frame:</b> {self.algorithm.frame_count} / {self.algorithm.frames_to_grab}<br>
        <br>
        <b>Current Color Statistics:</b><br>
        &nbsp;&nbsp;Brightness: {current_brightness:.1f}<br>
        &nbsp;&nbsp;Red: {current_red:.1f}<br>
        &nbsp;&nbsp;Green: {current_green:.1f}<br>
        &nbsp;&nbsp;Blue: {current_blue:.1f}<br>
        <br>
        <b>Trigger Enabled:</b> {self.algorithm.trigger_enabled}<br>
        """
        
        if self.algorithm.trigger_enabled:
            info += f"<b>Trigger Threshold:</b> {self.algorithm.trigger_color_threshold}<br>"
            info += f"<b>Stimulus Events:</b> {len(self.algorithm.stim_events)}<br>"
            info += f"<b>Cooldown:</b> {self.algorithm.cooldown_counter} frames<br>"
        
        self.info_text.setText(info)
    
    def process_events(self):
        """Process Qt events."""
        self.app.processEvents()
    
    def close(self):
        """Close visualizer."""
        self.window.close()
        logger.info("RGBVisualizer closed")
