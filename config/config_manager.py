"""
CLEF Configuration Manager

Centralized configuration loading, validation, and merging for CLEF platform.
Implements YAML + Pydantic validation strategy from refactor planning.

Author: Raymond Dunn
Version: 2.0.0
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from copy import deepcopy

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Configuration Models
# ============================================================================

class DeviceProperties(BaseModel):
    """Device-specific properties (e.g., camera exposure, laser power)."""
    model_config = ConfigDict(extra="allow")  # Allow arbitrary device-specific properties


class DeviceConfig(BaseModel):
    """Configuration for a single hardware device."""
    device_name: str = Field(..., description="Micro-Manager device name")
    device_type: str = Field(..., description="Device type (camera, stage, laser, etc.)")
    properties: DeviceProperties = Field(default_factory=DeviceProperties)
    configs: Dict[str, str] = Field(default_factory=dict, description="Micro-Manager config group presets (e.g., {'Channel': '488'})")


class ShutterConfig(BaseModel):
    """Shutter configuration."""
    device_name: str = Field(..., description="Shutter device name")
    state: bool = Field(..., description="Shutter open state (True=open, False=closed)")


class SystemProperties(BaseModel):
    """System-level Micro-Manager properties."""
    auto_shutter: Optional[bool] = Field(None, description="Auto shutter enabled")
    circular_buffer_mb: Optional[int] = Field(None, gt=0, description="Circular buffer size in MB")
    shutters: List[ShutterConfig] = Field(default_factory=list, description="Shutter states to set")
    
    model_config = ConfigDict(extra="allow")  # Allow additional system properties


class IlluminationChannel(BaseModel):
    """Configuration for an illumination channel."""
    name: str = Field(..., description="Channel name (e.g., '488nm', 'Brightfield')")
    device: str = Field(..., description="Device controlling this channel")
    wavelength: Optional[int] = Field(None, description="Wavelength in nm")
    power: Optional[float] = Field(None, description="Power setting (device-specific units)")
    exposure_ms: Optional[float] = Field(None, description="Exposure time in milliseconds")


class HardwareConfig(BaseModel):
    """Hardware configuration (hardware.yaml)."""
    backend: str = Field("pycromanager", description="Backend: pycromanager, pymmcore, or dummy")
    mm_config_path: Optional[str] = Field(None, description="Path to Micro-Manager .cfg file")
    stim_interface: str = Field("dummy", description="Stimulus interface class name")
    
    # Temporary field for backward compatibility (Phase 1-2)
    # Will be removed when hardware abstraction complete
    microscope_name: Optional[str] = Field(None, description="Microscope name (temporary)")
    
    devices: Dict[str, DeviceConfig] = Field(default_factory=dict)
    illumination_channels: List[IlluminationChannel] = Field(default_factory=list)
    
    # System-level Micro-Manager properties
    system_properties: Optional[SystemProperties] = Field(None, description="System-level MM settings")
    
    # Calibration data
    polygon_calibration_path: Optional[str] = Field(None, description="Path to polygon calibration JSON")
    
    # Strobe acquisition settings
    strobe_acquisition: bool = Field(False, description="Enable strobe illumination")
    strobe_inter_frame_interval_ms: int = Field(80, description="Strobe inter-frame interval (ms)")
    
    # Static ROI for stimulus
    use_static_stim_roi: bool = Field(False, description="Use static stimulus ROI")
    
    model_config = ConfigDict(extra="allow")  # Allow additional hardware-specific fields
    
    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v):
        allowed = ['pycromanager', 'pymmcore', 'dummy', 'test']
        if v not in allowed:
            raise ValueError(f"Backend must be one of {allowed}, got '{v}'")
        return v


class AcquisitionConfig(BaseModel):
    """Acquisition parameters."""
    num_frames: int = Field(100, gt=0, description="Number of frames to acquire")
    frame_rate: Optional[float] = Field(None, gt=0, description="Target frame rate (Hz)")
    
    # Z-stack settings
    z_stack: bool = Field(False, description="Enable Z-stack acquisition")
    z_planes: int = Field(1, gt=0, description="Number of Z planes")
    z_start: float = Field(0.0, description="Z-stack start position (µm)")
    z_end: float = Field(0.0, description="Z-stack end position (µm)")
    z_step: float = Field(1.0, gt=0, description="Z-stack step size (µm)")
    
    # Baseline and structural scan
    baseline_frames: int = Field(0, ge=0, description="Frames before stims allowed")
    save_structural_scan: str = Field("none", description="Structural scan type")
    
    @field_validator('z_end')
    @classmethod
    def validate_z_range(cls, v, info):
        values = info.data
        if values.get('z_stack') and values.get('z_planes', 1) > 1:
            if v <= values.get('z_start', 0):
                raise ValueError("z_end must be greater than z_start for z-stack acquisition")

        return v


class TreatmentDetails(BaseModel):
    """Treatment/condition details."""
    condition: str = Field("", description="Experimental condition")
    atr_concentration_uM: Optional[float] = Field(None, description="ATR concentration (µM)")
    
    model_config = ConfigDict(extra="allow")


class Orientation(BaseModel):
    """Anatomical orientation (domain-specific, e.g., C. elegans)."""
    nose: Optional[str] = Field(None, description="Nose orientation (e.g., left/right/other)")
    vnc: Optional[str] = Field(None, description="VNC orientation (e.g., up/down/other)")
    
    model_config = ConfigDict(extra="allow")


class SubjectMetadata(BaseModel):
    """Experimental subject metadata (generic, not worm-specific)."""
    subject_id: Optional[str] = Field(None, description="Subject identifier")
    subject_type: Optional[str] = Field(None, description="Subject type (e.g., 'C. elegans', 'cell culture')")
    genotype: Optional[str] = Field(None, description="Genetic background or strain")
    treatment: Optional[str] = Field(None, description="Experimental treatment (summary)")
    treatment_details: TreatmentDetails = Field(default_factory=TreatmentDetails)
    orientation: Orientation = Field(default_factory=Orientation)
    num_eggs: int = Field(0, ge=0, description="Number of eggs (for egg-laying organisms)")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    model_config = ConfigDict(extra="allow")  # Allow domain-specific fields


class DevOptions(BaseModel):
    """Development/testing options."""
    prefill_wb_ops: bool = Field(False, description="Prefill wboptions.mat and meta.mat")
    send_sms_on_completion: bool = Field(False, description="Send SMS when acquisition completes")
    
    model_config = ConfigDict(extra="allow")


class ExperimentConfig(BaseModel):
    """Experiment configuration (experiment.yaml)."""
    experiment_name: str = Field(..., description="Experiment name/identifier")
    experimenter: str = Field("Unknown", description="Experimenter name")
    output_dir: str = Field("./data", description="Output directory for data")
    save_images: bool = Field(True, description="Save acquired images")
    save_metadata: bool = Field(True, description="Save metadata JSON")
    save_mip_video: bool = Field(False, description="Save maximum intensity projection video")
    
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    subject: SubjectMetadata = Field(default_factory=SubjectMetadata)
    
    # Z-step size (duplicated for backward compatibility)
    z_step_size_um: float = Field(1.0, gt=0, description="Z-step size in micrometers")
    
    # Input recording for playback/simulation
    input_recording_path: Optional[str] = Field(None, description="Path to input TIFF for playback")
    
    # Development options
    dev_options: DevOptions = Field(default_factory=DevOptions)
    
    model_config = ConfigDict(extra="allow")


class AlgorithmParameters(BaseModel):
    """Algorithm-specific parameters."""
    # Threshold parameters (for derivative-based algorithms)
    stim_threshold_pos: float = Field(0.06, description="Positive threshold to trigger")
    stim_threshold_neg: float = Field(0.06, description="Negative threshold (abs value)")
    
    # Refractory period
    stim_cooldown_frames: int = Field(900, ge=0, description="Frames between allowed stims")
    
    # Stochastic stimulation
    skip_stimulation_probability: float = Field(0.1, ge=0.0, le=1.0, description="Prob of skipping stim")
    delay_stimulation_probability: float = Field(0.4, ge=0.0, le=1.0, description="Prob of delaying stim")
    stim_delay_frames_options: List[int] = Field(default_factory=lambda: [200, 400], description="Delay options")
    
    # Fixed timing (for StimOnsetFromList)
    stim_onset_list: List[int] = Field(default_factory=list, description="Fixed stim frame numbers")
    
    # PointAndClick specific
    stimulus_diameter_pixels: int = Field(10, gt=0, description="ROI diameter for point-and-click")
    
    model_config = ConfigDict(extra="allow")  # Allow arbitrary algorithm parameters


class StimulusParameters(BaseModel):
    """Stimulus-specific parameters."""
    enabled: bool = Field(False, description="Enable stimulus delivery")
    randomize: bool = Field(False, description="Use stochastic skip/delay")
    
    # Stimulus timing and intensity options
    duration_frames_options: List[int] = Field(default_factory=lambda: [48], description="Duration options (frames)")
    intensity_percent_options: List[int] = Field(default_factory=lambda: [10], description="Intensity options (0-100%)")
    
    # Selected values (from options)
    duration_frames: int = Field(48, gt=0, description="Actual duration used")
    intensity_percent: int = Field(10, ge=0, le=100, description="Actual intensity used")
    
    model_config = ConfigDict(extra="allow")


class AlgorithmConfig(BaseModel):
    """Algorithm configuration (algorithm.yaml)."""
    algorithm_type: str = Field("dummy", description="Algorithm class name")
    enable_gui: bool = Field(False, description="Enable algorithm GUI")
    gui_mode: str = Field("neural_imaging", description="GUI mode (neural_imaging/behavior)")
    save_algorithm_plot: bool = Field(False, description="Save algorithm output plots")
    
    algorithm_params: AlgorithmParameters = Field(default_factory=AlgorithmParameters)
    stimulus_params: StimulusParameters = Field(default_factory=StimulusParameters)
    
    model_config = ConfigDict(extra="allow")


class StimulusDeviceConfig(BaseModel):
    """
    Configuration for a specific stimulus device type.
    
    Defines how to control different stimulus hardware through Micro-Manager.
    """
    type: str = Field(..., description="Stimulus type: widefield_laser, polygon, led, dummy")
    
    # Widefield laser fields
    voltage_device: Optional[str] = Field(None, description="Voltage control device (e.g., DAC639)")
    voltage_property: Optional[str] = Field(None, description="Voltage property name (e.g., Volts)")
    ttl_device: Optional[str] = Field(None, description="TTL control device (e.g., TTL1-8)")
    ttl_line: Optional[str] = Field(None, description="TTL line name (e.g., TTL-4)")
    max_volts: Optional[float] = Field(None, description="Maximum voltage (for intensity scaling)")
    
    # Polygon/LED fields
    intensity_device: Optional[str] = Field(None, description="Intensity control device")
    intensity_property: Optional[str] = Field(None, description="Intensity property name")
    shutter_device: Optional[str] = Field(None, description="Shutter device name")
    slm_device: Optional[str] = Field(None, description="SLM device (polygon only, None = query MMC)")
    
    model_config = ConfigDict(extra="allow")  # Allow additional device-specific fields


class HardwareConfig(BaseModel):
    """Hardware configuration (hardware.yaml)."""
    backend: str = Field("pycromanager", description="Backend: pycromanager, pymmcore, or dummy")
    mm_config_path: Optional[str] = Field(None, description="Path to Micro-Manager .cfg file")
    stim_interface: str = Field("dummy", description="Stimulus interface class name")
    
    # Temporary field for backward compatibility (Phase 1-2)
    # Will be removed when hardware abstraction complete
    microscope_name: Optional[str] = Field(None, description="Microscope name (temporary)")
    
    # NEW: Stimulus device configurations
    stimulus_devices: Dict[str, StimulusDeviceConfig] = Field(
        default_factory=dict,
        description="Stimulus device configurations keyed by interface name"
    )
    
    devices: Dict[str, DeviceConfig] = Field(default_factory=dict)
    illumination_channels: List[IlluminationChannel] = Field(default_factory=list)
    
    # System-level Micro-Manager properties
    system_properties: Optional[SystemProperties] = Field(None, description="System-level MM settings")
    
    # Calibration data
    polygon_calibration_path: Optional[str] = Field(None, description="Path to polygon calibration JSON")
    
    # Strobe acquisition settings
    strobe_acquisition: bool = Field(False, description="Enable strobe illumination")
    strobe_inter_frame_interval_ms: int = Field(80, description="Strobe inter-frame interval (ms)")
    
    # Static ROI for stimulus
    use_static_stim_roi: bool = Field(False, description="Use static stimulus ROI")
    
    model_config = ConfigDict(extra="allow")  # Allow additional hardware-specific fields
    
    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v):
        allowed = ['pycromanager', 'pymmcore', 'dummy', 'test']
        if v not in allowed:
            raise ValueError(f"Backend must be one of {allowed}, got '{v}'")
        return v
    
    def get_stimulus_device_config(self, interface_name: Optional[str] = None) -> Optional[StimulusDeviceConfig]:
        """
        Get stimulus device configuration for a given interface.
        
        Args:
            interface_name: Stimulus interface name (uses self.stim_interface if None)
        
        Returns:
            StimulusDeviceConfig if found, None otherwise
        """
        interface_name = interface_name or self.stim_interface
        return self.stimulus_devices.get(interface_name)


# ============================================================================
# Configuration Manager
# ============================================================================

class ConfigManager:
    """
    Centralized configuration management for CLEF.
    
    Loads YAML configs, validates with Pydantic, merges defaults with user overrides.
    """
    
    def __init__(self, package_root: Optional[Path] = None):
        """
        Initialize ConfigManager.
        
        Args:
            package_root: Root directory of CLEF package (auto-detected if None)
        """
        if package_root is None:
            # Auto-detect: assume config_manager.py is in clef/config/
            package_root = Path(__file__).parent.parent
        
        self.package_root = Path(package_root)
        self.defaults_dir = self.package_root / "config" / "defaults"
        
        self.hardware_config: Optional[HardwareConfig] = None
        self.experiment_config: Optional[ExperimentConfig] = None
        self.algorithm_config: Optional[AlgorithmConfig] = None
        
        logger.info(f"ConfigManager initialized with package root: {self.package_root}")
    
    def load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load and parse YAML file."""
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            logger.debug(f"Loaded YAML from {path}")
            return data
        except FileNotFoundError:
            logger.error(f"Config file not found: {path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {path}: {e}")
            raise
    
    def merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two config dictionaries (override wins).
        
        Args:
            default: Base configuration dictionary
            override: User override dictionary
            
        Returns:
            Merged configuration dictionary
        """
        merged = deepcopy(default)
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                if key in merged and merged[key] != value:
                    logger.info(f"Overriding '{key}': {merged[key]} → {value}")
                merged[key] = value
        
        return merged
    
    def load_hardware_config(self, user_path: Optional[Path] = None) -> HardwareConfig:
        """
        Load hardware configuration.
        
        Args:
            user_path: Optional user override config path
            
        Returns:
            Validated HardwareConfig object
        """
        # Load package default
        default_path = self.defaults_dir / "hardware_default.yaml"
        default_data = self.load_yaml(default_path) if default_path.exists() else {}
        
        # Merge with user override if provided
        if user_path:
            user_data = self.load_yaml(Path(user_path))
            config_data = self.merge_configs(default_data, user_data)
        else:
            config_data = default_data
        
        # Validate with Pydantic
        try:
            self.hardware_config = HardwareConfig(**config_data)
            logger.info(f"Hardware config loaded: backend={self.hardware_config.backend}, "
                       f"stim={self.hardware_config.stim_interface}")
            return self.hardware_config
        except ValidationError as e:
            logger.error(f"Hardware config validation failed: {e}")
            raise
    
    def load_experiment_config(self, user_path: Optional[Path] = None) -> ExperimentConfig:
        """
        Load experiment configuration.
        
        Args:
            user_path: Optional user override config path
            
        Returns:
            Validated ExperimentConfig object
        """
        default_path = self.defaults_dir / "experiment_default.yaml"
        default_data = self.load_yaml(default_path) if default_path.exists() else {}
        
        if user_path:
            user_data = self.load_yaml(Path(user_path))
            config_data = self.merge_configs(default_data, user_data)
        else:
            config_data = default_data
        
        try:
            self.experiment_config = ExperimentConfig(**config_data)
            logger.info(f"Experiment config loaded: {self.experiment_config.experiment_name}")
            return self.experiment_config
        except ValidationError as e:
            logger.error(f"Experiment config validation failed: {e}")
            raise
    
    def load_algorithm_config(self, user_path: Optional[Path] = None) -> AlgorithmConfig:
        """
        Load algorithm configuration.
        
        Args:
            user_path: Optional user override config path
            
        Returns:
            Validated AlgorithmConfig object
        """
        default_path = self.defaults_dir / "algorithm_default.yaml"
        default_data = self.load_yaml(default_path) if default_path.exists() else {}
        
        if user_path:
            user_data = self.load_yaml(Path(user_path))
            config_data = self.merge_configs(default_data, user_data)
        else:
            config_data = default_data
        
        try:
            self.algorithm_config = AlgorithmConfig(**config_data)
            logger.info(f"Algorithm config loaded: {self.algorithm_config.algorithm_type}")
            return self.algorithm_config
        except ValidationError as e:
            logger.error(f"Algorithm config validation failed: {e}")
            raise
    
    def load_all_configs(self, 
                        hardware_path: Optional[Path] = None,
                        experiment_path: Optional[Path] = None,
                        algorithm_path: Optional[Path] = None):
        """
        Load all three configuration files.
        
        Args:
            hardware_path: Optional hardware config override
            experiment_path: Optional experiment config override
            algorithm_path: Optional algorithm config override
        """
        self.load_hardware_config(hardware_path)
        self.load_experiment_config(experiment_path)
        self.load_algorithm_config(algorithm_path)
        
        logger.info("All configurations loaded successfully")
    
    def validate_config(self) -> bool:
        """
        Validate loaded configurations for compatibility.
        
        Returns:
            True if all configs are valid and compatible
        """
        if not all([self.hardware_config, self.experiment_config, self.algorithm_config]):
            logger.error("Not all configs loaded - call load_all_configs() first")
            return False
        
        # Check backend/stim compatibility
        if self.hardware_config.backend == "dummy" or self.hardware_config.backend == "test":
            if self.hardware_config.stim_interface not in ["dummy", "no stim", "test"]:
                logger.warning(f"Backend is {self.hardware_config.backend} but stim_interface is "
                             f"'{self.hardware_config.stim_interface}' - may not work")
        
        # Check z-stack configuration consistency
        if self.experiment_config.acquisition.z_stack:
            if self.experiment_config.acquisition.z_planes <= 1:
                logger.warning("z_stack enabled but z_planes <= 1")
            if self.experiment_config.acquisition.z_step <= 0:
                logger.error("z_stack enabled but z_step <= 0")
                return False
        
        # Check stimulus parameters consistency
        if self.algorithm_config.stimulus_params.enabled:
            if not self.algorithm_config.stimulus_params.duration_frames_options:
                logger.warning("Stimulus enabled but no duration_frames_options specified")
            if not self.algorithm_config.stimulus_params.intensity_percent_options:
                logger.warning("Stimulus enabled but no intensity_percent_options specified")
        
        # Warn about microscope_name (temporary field)
        if self.hardware_config.microscope_name:
            logger.info(f"Using microscope_name: '{self.hardware_config.microscope_name}' "
                       "(temporary field, will be removed in Phase 2)")
        
        logger.info("Configuration validation passed")
        return True