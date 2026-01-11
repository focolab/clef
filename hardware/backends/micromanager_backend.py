"""
Micro-Manager hardware backend.

Wraps Micro-Manager (MMC) API calls to provide unified hardware interface.
Supports both pycromanager and pymmcore apis.
"""

import logging, json
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

from hardware.backends.base_backend import BaseHardwareBackend
from hardware.camera_interface import CameraInterface
from hardware.stage_interface import StageInterface
from hardware.stimulus_interface import StimulusInterface
from hardware.projector_interface import ProjectorInterface
from config.config_manager import HardwareConfig

        
# Import utilities for mask generation
from utils import wbliveUtils
from utils import numba_utils

logger = logging.getLogger(__name__)

# Import MMSubroutines to use existing initialization logic
from utils import MMSubroutines

# Import JavaObject for pycromanager ASI stage buffer
try:
    from pycromanager import JavaObject
except ImportError:
    # Fallback if pycromanager not available
    JavaObject = None
        
class MicroManagerCamera(CameraInterface):
    """Camera interface wrapping Micro-Manager Core."""
    
    def __init__(self, mmc, backend: str, roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize Micro-Manager camera interface.
        
        Args:
            mmc: Micro-Manager Core object
            backend: Backend type ('pycromanager' or 'pymmcore')
            roi: Optional initial ROI (x, y, width, height)
        """
        self.mmc = mmc
        self.backend = backend
        
        # Get initial ROI
        if roi:
            self._roi = roi
            self.mmc.setROI(roi[0], roi[1], roi[2], roi[3])
        else:
            roi_obj = self.mmc.getROI()
            if backend == "pycromanager":
                # pycromanager returns Java objects
                self._roi = (roi_obj.getX(), roi_obj.getY(), 
                           roi_obj.getWidth(), roi_obj.getHeight())
            else:
                # pymmcore returns tuple
                self._roi = roi_obj
        
        self._width = self._roi[2]
        self._height = self._roi[3]
    
    def acquire_frame(self) -> np.ndarray:
        """Acquire a single frame."""
        self.mmc.snapImage()
        return self.get_image()
    
    def start_acquisition(self, buffer_size: int = 0) -> None:
        """Start continuous sequence acquisition."""
        if buffer_size != 0:
            logger.warning('Attempting to set camera buffer with {}, but overwriting!')
        self.mmc.setCircularBufferMemoryFootprint(10000)
        self.mmc.startContinuousSequenceAcquisition(0)
        logger.debug("Micro-Manager: Started continuous acquisition")
    
    def stop_acquisition(self) -> None:
        """Stop continuous sequence acquisition."""
        self.mmc.stopSequenceAcquisition()
        logger.debug("Micro-Manager: Stopped acquisition")
    
    def get_remaining_image_count(self) -> int:
        """Get remaining images in buffer."""
        return self.mmc.getRemainingImageCount()
    
    def pop_next_image(self) -> np.ndarray:
        """Pop next image from buffer."""
        img = self.mmc.popNextImage().astype(np.uint16)
        if self.backend == "pycromanager":
            img = img.reshape((self._height, self._width))
        return img
    
    def snap_image(self) -> None:
        """Trigger single snap."""
        self.mmc.snapImage()
    
    def get_image(self) -> np.ndarray:
        """Get most recent image."""
        img = self.mmc.getImage().astype(np.uint16)
        if self.backend == "pycromanager":
            img = img.reshape((self._height, self._width))
        return img
    
    def clear_buffer(self) -> None:
        """Clear circular buffer."""
        self.mmc.clearCircularBuffer()
        logger.debug("Micro-Manager: Cleared buffer")
    
    def set_exposure(self, exposure_ms: float) -> None:
        """Set exposure time."""
        self.mmc.setExposure(exposure_ms)
        logger.debug(f"Micro-Manager: Set exposure to {exposure_ms} ms")
    
    def get_exposure(self) -> float:
        """Get exposure time."""
        return self.mmc.getExposure()
    
    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set ROI."""
        self.mmc.setROI(x, y, width, height)
        self._roi = (x, y, width, height)
        self._width = width
        self._height = height
        logger.debug(f"Micro-Manager: Set ROI to ({x}, {y}, {width}, {height})")
    
    def get_roi(self) -> Tuple[int, int, int, int]:
        """Get ROI."""
        roi_obj = self.mmc.getROI()
        if self.backend == "pycromanager":
            return (roi_obj.getX(), roi_obj.getY(), 
                   roi_obj.getWidth(), roi_obj.getHeight())
        else:
            return roi_obj
    
    def get_image_size(self) -> Tuple[int, int]:
        """Get image size."""
        return (self._width, self._height)
    
    def configure_camera(self, config: Dict[str, Any]) -> None:
        """
        Configure camera settings.
        
        Args:
            config: Dictionary containing camera configuration parameters
                   (e.g., exposure, binning, properties)
        """

        # TODO config will be an ExperimentConfig...
        
        # Set exposure if provided
        if "exposure" in config:
            self.set_exposure(config["exposure"])
        
        # Set ROI if provided
        if "roi" in config:
            roi = config["roi"]
            if isinstance(roi, (tuple, list)) and len(roi) == 4:
                self.set_roi(roi[0], roi[1], roi[2], roi[3])
        
        # Set device properties if provided
        if "properties" in config:
            cam_device = self.mmc.getCameraDevice()
            for prop_name, prop_value in config["properties"].items():
                try:
                    self.mmc.setProperty(cam_device, prop_name, prop_value)
                except Exception as e:
                    logger.warning(f"Could not set property {prop_name}={prop_value}: {e}")
        
        logger.debug(f"Micro-Manager: Camera configured with {config}")


class MicroManagerStage(StageInterface):
    """Stage interface wrapping Micro-Manager Core."""
    
    def __init__(self, mmc, backend: str):
        """
        Initialize Micro-Manager stage interface.
        
        Args:
            mmc: Micro-Manager Core object
            backend: Backend type ('pycromanager' or 'pymmcore')
        """
        self.mmc = mmc
        self.backend = backend
        self._focus_device = None
    
    def get_position(self, axis: Optional[str] = None) -> Union[float, Tuple[float, ...]]:
        """Get stage position."""
        if axis == 'Z' or axis is None:
            focus_dev = self.get_focus_device_name()
            pos = self.mmc.getPosition(focus_dev)
            if axis == 'Z':
                return pos
            return (pos,)
        else:
            # For X/Y stages, would need to get XYStage device
            # For now, just return Z position
            focus_dev = self.get_focus_device_name()
            return (self.mmc.getPosition(focus_dev),)
    
    def move_to_position(self, position: Union[float, Tuple[float, ...]], axis: Optional[str] = None) -> None:
        """Move stage to position."""
        focus_dev = self.get_focus_device_name()
        if isinstance(position, (int, float)):
            self.mmc.setPosition(focus_dev, float(position))
        elif isinstance(position, (tuple, list)) and len(position) > 0:
            self.mmc.setPosition(focus_dev, float(position[0]))
        logger.debug(f"Micro-Manager: Moved stage to {position}")
    
    def run_z_stack(self, z_start: float, z_end: float, z_step: float, num_planes: int) -> None:
        """
        Configure Z-stack sequence.
        
        This method configures the stage for Z-stack acquisition using ASI stage buffer
        for pycromanager backend. For other backends, it may use different methods.
        """
        # Use configure_stage to set up ASI stage buffer
        config = {
            "z_start": z_start,
            "z_end": z_end,
            "z_step": z_step,
            "pad_z": 0,
        }
        self.configure_stage(config)
        logger.debug(f"Micro-Manager: Z-stack from {z_start} to {z_end}, step {z_step}, {num_planes} planes")
    
    def configure_stage(self, config: Dict[str, Any]) -> None:
        """
        Configure stage for acquisition sequences (e.g., ASI stage buffer).
        
        This method sets up ASI stage sequences with Z positions and optional
        TTL property sequences for synchronized acquisition.
        
        Args:
            config: Dictionary containing stage configuration parameters:
                   - z_start: Starting Z position (float, required)
                   - z_end: Ending Z position (float, required)
                   - z_step: Step size (float, required)
                   - pad_z: Number of padding steps at start (int, default 0)
                   - ttl_device: TTL device name for property sequences (str, default "TTL1-8")
                   - ttl_state: TTL state value for each step (str, default "18")
        """
        z_start = config.get("z_start")
        z_end = config.get("z_end")
        z_step = config.get("z_step")
        pad_z = config.get("pad_z", 0)
        ttl_device = config.get("ttl_device", "TTL1-8")
        ttl_state = config.get("ttl_state", "18")
        
        if z_start is None or z_end is None or z_step is None:
            raise ValueError("z_start, z_end, and z_step are required for stage configuration")
        
        # Get focus device
        stage = self.get_focus_device_name()
        
        # Quick semantic check for case of 1Z plane imaging + structural scan
        # spec loop will hang unless zStepSize is set to some value > 0
        if z_step == 0:
            z_step = 1
            logger.warning("z_step was 0, setting to 1 to avoid hanging")
        
        # For pycromanager backend, use JavaObject for stage sequences
        if self.backend == "pycromanager":
            if JavaObject is None:
                raise ImportError("pycromanager.JavaObject required for ASI stage buffer configuration")
            
            # Create Java objects for stage sequence
            dv = JavaObject("mmcorej.DoubleVector")
            sv = JavaObject("mmcorej.StrVector")
            dv_list = []
            
            z = z_start
            
            # Pad zstep array with extra steps at zStart
            for i in range(pad_z):
                dv.add(z)
                dv_list.append(z)
                sv.add("0")
            
            # Ascending z-steps
            while z <= z_end:
                dv.add(z)
                dv_list.append(z)
                sv.add(ttl_state)
                z += z_step
            
            logger.info(
                f"Uploading {len(dv_list)} zPositions to ASI stage: {dv_list}"
            )
            
            # Upload and configure
            self.mmc.setPosition(stage, z_start)
            self.mmc.waitForDevice(stage)
            self.mmc.stopStageSequence(stage)
            self.mmc.loadStageSequence(stage, dv)
            self.mmc.stopPropertySequence(ttl_device, "State")
            self.mmc.loadPropertySequence(ttl_device, "State", sv)
            self.mmc.startStageSequence(stage)
            self.mmc.startPropertySequence(ttl_device, "State")
            
            logger.debug(f"ASI stage buffer configured: {len(dv_list)} positions from {z_start} to {z_end}")
        
        else:
            # For pymmcore backend, ASI stage buffer may not be available
            # Log a warning and use basic Z-stack configuration
            logger.warning(
                f"ASI stage buffer configuration not fully supported for {self.backend} backend. "
                f"Using basic Z-stack configuration."
            )
            # Basic implementation: just set position range
            self.mmc.setPosition(stage, z_start)
            self.mmc.waitForDevice(stage)
    
    def stop_sequence(self) -> None:
        """Stop stage sequence."""
        focus_dev = self.get_focus_device_name()
        self.mmc.stopStageSequence(focus_dev)
        logger.debug("Micro-Manager: Stopped stage sequence")
    
    def wait_for_device(self, timeout_ms: Optional[int] = None) -> None:
        """Wait for stage movement."""
        focus_dev = self.get_focus_device_name()
        self.mmc.waitForDevice(focus_dev)
    
    def get_focus_device_name(self) -> str:
        """Get focus device name."""
        if self._focus_device is None:
            self._focus_device = self.mmc.getFocusDevice()
        return self._focus_device


class MicroManagerStimulus(StimulusInterface):
    """
    Stimulus interface wrapping Micro-Manager Core.
    
    Supports multiple stimulus types with device mappings from configuration:
    - Widefield laser (e.g., InvCore-SpinningDisk-639)
    - Polygon/LDI (e.g., InvCore-LDI-Polygon-640)
    - LED (e.g., InvCore-ThunderscopeLED3)
    """
    
    def __init__(self, mmc, backend: str, hardware_config: Optional[HardwareConfig]):
        """
        Initialize Micro-Manager stimulus interface.
        
        Args:
            mmc: Micro-Manager Core object
            backend: Backend type ('pycromanager' or 'pymmcore')
            hardware_config: HardwareConfig object containing stimulus device configs
        """
        self.mmc = mmc
        self.backend = backend
        self.hardware_config = hardware_config
        self._active = False
        self._configured = False
        self._stim_type = None
        self._device_config = None
        self._current_params = None
        
        # Polygon-specific attributes
        self.slm_device = None
        self.polygon_dims = None
        self.calibration_points = None
    
    def configure_stimulus(self, config: Dict[str, Any]) -> None:
        """
        Configure stimulus hardware settings.
        
        Args:
            config: Dictionary containing:
                   - interface_type: stimulus interface name
                   - intensity: default intensity
                   - calibration_path: optional calibration file path
        """
        interface_type = config.get("interface_type", "")
        
        # Get device configuration from HardwareConfig
        self._device_config = self.hardware_config.get_stimulus_device_config(interface_type)
        
        if self._device_config is None:
            logger.warning(f"No stimulus device config found for: {interface_type}")
            return
        
        self._stim_type = interface_type
        stim_type = self._device_config.type
        
        logger.info(f"Configuring stimulus for {interface_type} (type: {stim_type})")
        
        # Type-specific configuration
        if stim_type == "widefield_laser":
            self._configure_widefield_laser(config)
        elif stim_type == "polygon":
            self._configure_polygon(config)
        elif stim_type == "led":
            self._configure_led(config)
        elif stim_type == "dummy":
            logger.info("Dummy stimulus configured (no hardware operations)")
        else:
            logger.warning(f"Unknown stimulus type: {stim_type}")
            return
        
        self._configured = True
        logger.info(f"Stimulus configured for {interface_type}")
    
    def _configure_widefield_laser(self, config: Dict[str, Any]) -> None:
        """Configure widefield laser stimulus."""
        dev = self._device_config
        
        # Set TTL line high to enable
        if dev.ttl_device and dev.ttl_line:
            try:
                self.mmc.setProperty(dev.ttl_device, dev.ttl_line, 1)
                logger.debug(f"Set {dev.ttl_device}.{dev.ttl_line} = 1")
            except Exception as e:
                logger.warning(f"Could not set TTL line: {e}")
        
        # Initialize voltage to 0
        if dev.voltage_device:
            try:
                voltage_prop = dev.voltage_property or "Volts"
                self.mmc.setProperty(dev.voltage_device, voltage_prop, 0)
                logger.debug(f"Initialized {dev.voltage_device}.{voltage_prop} to 0V")
            except Exception as e:
                logger.warning(f"Could not initialize voltage: {e}")
    
    def _configure_polygon(self, config: Dict[str, Any]) -> None:
        """Configure polygon/LDI stimulus."""
        dev = self._device_config
        logger.debug(f'Configuring polygon device with config {dev}')
        
        # Get SLM device
        try:
            if dev.slm_device:
                self.slm_device = dev.slm_device
            else:
                self.slm_device = self.mmc.getSLMDevice()
            
            self.mmc.setSLMDevice(self.slm_device)
            
            # Get polygon dimensions
            width = self.mmc.getSLMWidth(self.slm_device)
            height = self.mmc.getSLMHeight(self.slm_device)
            self.polygon_dims = (width, height)
            
            logger.info(f"Polygon SLM: {self.slm_device}, dimensions: {self.polygon_dims}")
        except Exception as e:
            logger.error(f"Could not configure SLM device: {e}")
            return
        
        # # Set config group for simultaneous imaging (if using pymmcore)
        # if self.backend == "pymmcore":
        #     try:
        #         self.mmc.setConfig("Mightex-Setup", "640-SP")
        #     except Exception as e:
        #         logger.warning(f"Could not set Mightex config: {e}")
        
        # Initialize LDI off but open shutter
        try:
            if dev.intensity_device and dev.intensity_property:
                self.mmc.setProperty(dev.intensity_device, dev.intensity_property, 0)
            
            if self.slm_device:
                self.mmc.setSLMPixelsTo(self.slm_device, 0)
            
            if dev.shutter_device:
                self.mmc.setShutterOpen(dev.shutter_device, True)
            
            logger.debug("Initialized polygon: intensity=0, shutter open, SLM blank")
        except Exception as e:
            logger.warning(f"Could not initialize polygon: {e}")
        
        # Load calibration points
        calibration_path = config.get("calibration_path") or self.hardware_config.polygon_calibration_path
        if calibration_path:
            self.calibration_points = self._load_polygon_calibration(calibration_path)
        else:
            # Try default path
            default_path = "./res/peripherals/Mightex Polygon P1000/calibrations.json"
            logger.warning(f'No polygon calibration path provided, falling back on default at {default_path}')
            self.calibration_points = self._load_polygon_calibration(default_path)
            if self.calibration_points is None:
                logger.error(f'No calibration points detected for Polygon, behavior may be undefined.')
    
    def _configure_led(self, config: Dict[str, Any]) -> None:
        """Configure LED stimulus."""
        dev = self._device_config
        
        # Initialize LED off but open shutter
        try:
            if dev.intensity_device and dev.intensity_property:
                self.mmc.setProperty(dev.intensity_device, dev.intensity_property, 0)
            
            if dev.shutter_device:
                self.mmc.setShutterOpen(dev.shutter_device, True)
            
            logger.debug("Initialized LED: intensity=0, shutter open")
        except Exception as e:
            logger.warning(f"Could not initialize LED: {e}")
    
    def activate_stimulus(self, params: Dict[str, Any]) -> None:
        """
        Activate stimulus with given parameters.
        
        Args:
            params: Dictionary containing:
                   - intensity: intensity value (0-100 for %, or device-specific)
                   - position: optional (x, y) for polygon
                   - diameter: optional diameter for polygon
        """
        if not self._configured or self._device_config is None:
            logger.warning(f"Stimulus not configured: {self._configured}, or device config not set: {self._device_config}, activation may be undefined.")
        
        self._active = True
        self._current_params = params

        stim_type = self._device_config.type
        intensity = params.get("intensity", 10)
        
        if stim_type == "widefield_laser":
            self._activate_widefield_laser(intensity)
        elif stim_type == "polygon":
            self._activate_polygon(intensity)
        elif stim_type == "led":
            self._activate_led(intensity)
        elif stim_type == "dummy":
            logger.debug(f"Dummy stimulus activated: intensity={intensity}")
        else:
            logger.warning("No valid stim type found for activation.")
    
    def _activate_widefield_laser(self, intensity: float) -> None:
        """Activate widefield laser."""
        dev = self._device_config
        
        # Convert intensity percent to volts
        if dev.max_volts is None:
            logger.error("max_volts not configured for widefield laser")
            return
        
        volts = (intensity / 100.0) * dev.max_volts
        voltage_prop = dev.voltage_property or "Volts"
        
        try:
            self.mmc.setProperty(dev.voltage_device, voltage_prop, round(volts, 2))
            logger.debug(f"Activated widefield laser: {volts:.2f}V ({intensity}%)")
        except Exception as e:
            logger.error(f"Could not activate widefield laser: {e}")
    
    def _activate_polygon(self, intensity: float) -> None:
        """Activate polygon stimulus."""
        dev = self._device_config
        
        try:
            self.mmc.setProperty(
                dev.intensity_device,
                dev.intensity_property,
                int(intensity)
            )
            logger.debug(f"Activated polygon: intensity={intensity}")
        except Exception as e:
            logger.error(f"Could not activate polygon: {e}")
    
    def _activate_led(self, intensity: float) -> None:
        """Activate LED stimulus."""
        dev = self._device_config
        
        try:
            self.mmc.setProperty(
                dev.intensity_device,
                dev.intensity_property,
                int(intensity)
            )
            logger.debug(f"Activated LED: intensity={intensity}")
        except Exception as e:
            logger.error(f"Could not activate LED: {e}")
    
    def deactivate_stimulus(self) -> None:
        """Deactivate currently active stimulus."""
        if not self._configured or self._device_config is None:
            return
        
        stim_type = self._device_config.type
        
        if stim_type == "widefield_laser":
            self._deactivate_widefield_laser()
        elif stim_type == "polygon":
            self._deactivate_polygon()
        elif stim_type == "led":
            self._deactivate_led()
        elif stim_type == "dummy":
            logger.debug("Dummy stimulus deactivated")
        
        self._active = False
        self._current_params = None
    
    def _deactivate_widefield_laser(self) -> None:
        """Deactivate widefield laser."""
        dev = self._device_config
        voltage_prop = dev.voltage_property or "Volts"
        
        try:
            self.mmc.setProperty(dev.voltage_device, voltage_prop, 0)
            logger.debug("Deactivated widefield laser")
        except Exception as e:
            logger.error(f"Could not deactivate widefield laser: {e}")
    
    def _deactivate_polygon(self) -> None:
        """Deactivate polygon stimulus."""
        dev = self._device_config
        
        try:
            self.mmc.setProperty(dev.intensity_device, dev.intensity_property, 0)
            logger.debug("Deactivated polygon")
        except Exception as e:
            logger.error(f"Could not deactivate polygon: {e}")
    
    def _deactivate_led(self) -> None:
        """Deactivate LED stimulus."""
        dev = self._device_config
        
        try:
            self.mmc.setProperty(dev.intensity_device, dev.intensity_property, 0)
            logger.debug("Deactivated LED")
        except Exception as e:
            logger.error(f"Could not deactivate LED: {e}")
    
    def is_stimulus_active(self) -> bool:
        """Check if stimulus is currently active."""
        return self._active
    
    def update_polygon_mask(self, stim_params: Dict[str, Any]) -> None:
        """
        Update polygon mask based on stimulus parameters.
        
        This is polygon-specific and generates/uploads masks to the SLM.
        
        Args:
            stim_params: Dictionary containing event with mask parameters
        """
        if self._device_config is None or self._device_config.type != "polygon":
            logger.warning("update_polygon_mask called on non-polygon stimulus")
            return
        
        if self.slm_device is None:
            logger.error("SLM device not configured")
            return
        
        event = stim_params.get('event', {})
        event_type = event.get('event_type')
        
        # Get calibration and ROI
        if self.calibration_points is None:
            logger.error("Calibration points not loaded")
            return
        
        pcx = self.calibration_points['pcx']
        pcy = self.calibration_points['pcy']
        icx = self.calibration_points['icx']
        icy = self.calibration_points['icy']
        
        # Get ROI from config or params
        roi = stim_params.get('roi', [0, 0])
        width, height = self.polygon_dims
        
        # Generate mask based on event type
        if event_type in ['circle-click', 'circle-button', 'hammer-of-dawn']:
            cx = int(event["x"])
            cy = int(event["y"])
            diameter = int(event.get("stim_diameter", 20))
            
            mask = numba_utils.generate_pg_ellipse_mask(
                cx, cy, pcx, pcy, icx, icy,
                diameter, roi[0], roi[1], width, height
            )
        
        elif event_type in ['pulse-rect-roi-list', 'stream-rect-roi-list']:
            stim_rect_roi_list = event.get('stim_rect_roi_list', {})
            x_list = np.array(stim_rect_roi_list.get('x', []))
            y_list = np.array(stim_rect_roi_list.get('y', []))
            width_list = np.array(stim_rect_roi_list.get('width', []))
            height_list = np.array(stim_rect_roi_list.get('height', []))
            
            mask = numba_utils.generate_pg_multi_rectangle_mask(
                x_list, y_list, width_list, height_list,
                pcx, pcy, icx, icy,
                roi[0], roi[1], width, height
            )
        
        elif event_type == 'full-field-button':
            mask = np.ones(shape=self.polygon_dims, dtype=np.uint8) * 255
            self.mmc.setSLMPixelsTo(self.slm_device, 255)
            logger.debug("Set polygon to full-field mask")
            return
        
        else:
            logger.warning(f"Unknown polygon event type: {event_type}")
            return
        
        # Upload mask to SLM
        mask = mask * 255  # mask is uint8, values 1-255 specify dithering
        self.mmc.setSLMImage(self.slm_device, mask.astype(np.uint8).flatten())
        logger.debug(f"Updated polygon mask for {event_type}")
    
    def get_polygon_dimensions(self) -> Optional[Tuple[int, int]]:
        """Get polygon SLM dimensions."""
        return self.polygon_dims
    
    def get_calibration_points(self) -> Optional[Dict[str, Any]]:
        """Get polygon calibration points."""
        return self.calibration_points
    
    def _load_polygon_calibration(self, calibration_fname: str) -> Optional[Dict[str, Any]]:
        """
        Load calibration points for polygon from JSON file.
        
        Args:
            calibration_fname: Path to calibration JSON file
        
        Returns:
            Dictionary containing pcx, pcy, icx, icy arrays
        """
        try:
            with open(calibration_fname) as f:
                md = json.load(f)
                calibrations = md["calibrations"]
                
                # Get current objective and binning
                obj = self.mmc.getProperty("ObjectiveTurret", "Label")
                cam = self.mmc.getCameraDevice()
                binning = self.mmc.getProperty(cam, "Binning")
                
                # Find matching calibrations
                dt_list = []
                for cali in calibrations:
                    if cali["objective"] == obj and cali["binning"] == binning:
                        dt_list.append(cali["datetime"])
                
                if not dt_list:
                    logger.error(
                        f"No matching calibration found for objective: {obj}, binning: {binning}"
                    )
                    return None
                
                # Get latest calibration
                dt_list.sort()
                latest_dt = dt_list[-1]
                
                # Find and return calibration data
                for cali in calibrations:
                    if (cali["objective"] == obj and 
                        cali["datetime"] == latest_dt):
                        
                        logger.info(f'Loaded calibration: {cali}')
                        
                        return {
                            "pcx": np.array(cali["pcx"]),
                            "pcy": np.array(cali["pcy"]),
                            "icx": np.array(cali["icx"]),
                            "icy": np.array(cali["icy"])
                        }
        
        except Exception as e:
            logger.error(f"Error loading calibration points: {e}")
            return None
        

class MicroManagerProjector(ProjectorInterface):
    """Projector/SLM interface wrapping Micro-Manager Core."""
    
    def __init__(self, mmc, backend: str, device_name: Optional[str] = None):
        """
        Initialize Micro-Manager projector interface.
        
        Args:
            mmc: Micro-Manager Core object
            backend: Backend type ('pycromanager' or 'pymmcore')
            device_name: Optional SLM device name (None = query from MMC)
        """
        self.mmc = mmc
        self.backend = backend
        
        # Get SLM device
        if device_name:
            self._device_name = device_name
        else:
            self._device_name = self.mmc.getSLMDevice()
        
        self.mmc.setSLMDevice(self._device_name)
        
        # Get dimensions
        self._width = self.mmc.getSLMWidth(self._device_name)
        self._height = self.mmc.getSLMHeight(self._device_name)
        
        logger.debug(f"Initialized projector: {self._device_name}, "
                    f"dimensions: ({self._width}, {self._height})")
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get projector dimensions."""
        return (self._width, self._height)
    
    def set_image(self, image: np.ndarray) -> None:
        """
        Set projector image/mask.
        
        Args:
            image: 2D numpy array (uint8), shape (height, width)
        """
        if image.shape != (self._height, self._width):
            raise ValueError(
                f"Image shape {image.shape} doesn't match projector "
                f"dimensions ({self._height}, {self._width})"
            )
        
        # Flatten and upload to SLM
        self.mmc.setSLMImage(self._device_name, image.astype(np.uint8).flatten())
        logger.debug(f"Updated projector image: {self._device_name}")
    
    def set_pixels_to(self, value: int) -> None:
        """Set all pixels to uniform value."""
        if not 0 <= value <= 255:
            raise ValueError(f"Pixel value must be 0-255, got {value}")
        
        self.mmc.setSLMPixelsTo(self._device_name, value)
        logger.debug(f"Set projector pixels to {value}: {self._device_name}")
    
    def get_device_name(self) -> str:
        """Get projector device name."""
        return self._device_name    
        

class MicroManagerBackend(BaseHardwareBackend):
    """
    Micro-Manager hardware backend.
    
    Wraps Micro-Manager Core API to provide unified hardware interface.
    Supports both pycromanager and pymmcore backends.
    """
    
    def __init__(self, config: HardwareConfig):
        """Initialize Micro-Manager backend."""
        super().__init__(config)
        self.mmc = None
        self._projector: Optional[ProjectorInterface] = None

    def initialize(self, input_recording: Optional[str] = None) -> None:
        """
        Initialize Micro-Manager hardware.
        
        Args:
            input_recording: Optional path to input recording for playback mode
        """
        logger.info(f"Initializing Micro-Manager backend: {self.config.backend}")

        # Build args dict for MMSubroutines (temporary bridge)
        # This will be removed when MMSubroutines is fully refactored
        args = {
            "gooey_args": {
                "acquisition_backend": self.config.backend,
                "microscope_name": self.config.microscope_name or "unknown",
                "input_recording": input_recording,
            }
        }
        
        # Use config file from HardwareConfig
        config_file = self.config.mm_config_path
        
        # Initialize MMC using existing function
        self.mmc = MMSubroutines.initialize_mmc(args, config_file=config_file)
        
        # Get ROI
        roi_obj = self.mmc.getROI()
        if self.config.backend == "pycromanager":
            roi = (roi_obj.getX(), roi_obj.getY(), 
                  roi_obj.getWidth(), roi_obj.getHeight())
        else:
            roi = roi_obj
        
        # Create interfaces
        self._camera = MicroManagerCamera(self.mmc, self.config.backend, roi)
        self._stage = MicroManagerStage(self.mmc, self.config.backend)
        self._stimulus = MicroManagerStimulus(self.mmc, self.config.backend, self.config)

        # Create projector interface if needed (for polygon stimulus)
        if self.config.stim_interface and "polygon" in self.config.stim_interface.lower():
            # Get SLM device name from stimulus config
            stim_config = self.config.get_stimulus_device_config()
            slm_device = stim_config.slm_device if stim_config else None
            self._projector = MicroManagerProjector(self.mmc, self.config.backend, slm_device)
            logger.debug("Created projector interface for polygon stimulus")

        # Apply device properties and system properties from config
        self._apply_device_properties()
        self._apply_device_configs()
        self._apply_system_properties()

        # Apply initial component configuration -- could happen here?
        # self._camera.configure_camera()
        # self._stage.configure_stage()
        # self._stimulus.configure_stimulus()
        
        self._initialized = True
        logger.info("Micro-Manager backend initialized")
    
    def close(self) -> None:
        """Close Micro-Manager hardware."""
        logger.info("Closing Micro-Manager backend...")
        
        if self._camera:
            self._camera.stop_acquisition()
        
        # Use MMSubroutines.close for cleanup
        if self.mmc:
            # Build minimal args for close function
            args = {
                "gooey_args": {
                    "microscope_name": self.config.microscope_name or "unknown",
                }
            }
            MMSubroutines.close(self.mmc, args)
        
        self._initialized = False
        logger.info("Micro-Manager backend closed")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get hardware metadata."""
        if not self._initialized or not self.mmc:
            return {}
        
        # Use MMSubroutines.get_metadata
        args = {
            "gooey_args": {
                "microscope_name": self.config.microscope_name or "unknown",
            }
        }
        metadata = MMSubroutines.get_metadata(args, self.mmc)
        metadata['backend'] = self.config.backend
        return metadata
    
    def get_mmc(self):
        """
        Get underlying Micro-Manager Core object.
        
        This method provides access to the raw MMC object for components
        that haven't been refactored yet. This is temporary and will be
        removed as components are migrated to use the hardware abstraction.
        
        Returns:
            Micro-Manager Core object
        """
        return self.mmc
    
    def _apply_device_properties(self) -> None:
        """
        Apply device-specific properties from config.
        
        Sets properties using mmc.setProperty() for each device configured
        in HardwareConfig.devices.
        """
        if not self.config.devices:
            return
        
        for device_key, device_config in self.config.devices.items():
            device_name = device_config.device_name
            properties = device_config.properties
            
            # Convert Pydantic model to dict (handles extra="allow" fields)
            if hasattr(properties, 'model_dump'):
                props_dict = properties.model_dump(exclude_unset=True)
            else:
                props_dict = dict(properties) if hasattr(properties, '__dict__') else {}
            
            for prop_name, prop_value in props_dict.items():
                try:
                    self.mmc.setProperty(device_name, prop_name, prop_value)
                    logger.debug(f"Set {device_name}.{prop_name} = {prop_value}")
                except Exception as e:
                    logger.warning(
                        f"Could not set {device_name}.{prop_name} = {prop_value}: {e}"
                    )
    
    def _apply_device_configs(self) -> None:
        """
        Apply device config group presets from config.
        
        Sets Micro-Manager config group presets using mmc.setConfig() for each
        device configured in HardwareConfig.devices.
        """
        if not self.config.devices:
            return
        
        for device_key, device_config in self.config.devices.items():
            configs = device_config.configs
            
            for config_group, preset_name in configs.items():
                try:
                    self.mmc.setConfig(config_group, preset_name)
                    logger.debug(f"Set config {config_group} = {preset_name}")
                except Exception as e:
                    logger.warning(
                        f"Could not set config {config_group} = {preset_name}: {e}"
                    )
    
    def _apply_system_properties(self) -> None:
        """
        Apply system-level properties from config.
        
        Sets system-level Micro-Manager settings like auto_shutter,
        circular_buffer_memory_footprint, and shutter states.
        """
        if not self.config.system_properties:
            return
        
        sys_props = self.config.system_properties
        
        # Auto shutter
        if sys_props.auto_shutter is not None:
            try:
                self.mmc.setAutoShutter(sys_props.auto_shutter)
                logger.debug(f"Set auto_shutter = {sys_props.auto_shutter}")
            except Exception as e:
                logger.warning(f"Could not set auto_shutter: {e}")
        
        # Circular buffer
        if sys_props.circular_buffer_mb is not None:
            try:
                # Note: setCircularBufferMemoryFootprint unit may vary by Micro-Manager version
                # Existing code uses values like 10000 directly. We pass MB value as-is to match
                # existing behavior. If your Micro-Manager version expects bytes, multiply by 1024*1024.
                # self.mmc.setCircularBufferMemoryFootprint(sys_props.circular_buffer_mb)
                self.mmc.setCircularBufferMemoryFootprint(10000)
                logger.debug(f"Set circular_buffer_memory_footprint = {sys_props.circular_buffer_mb}")
            except Exception as e:
                logger.warning(f"Could not set circular_buffer_memory_footprint: {e}")
        
        # Shutters
        for shutter_config in sys_props.shutters:
            try:
                self.mmc.setShutterOpen(shutter_config.device_name, shutter_config.state)
                logger.debug(
                    f"Set shutter {shutter_config.device_name} = {'open' if shutter_config.state else 'closed'}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not set shutter {shutter_config.device_name}: {e}"
                )

