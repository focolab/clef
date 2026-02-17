"""
CLEF Configuration Manager

Centralized configuration loading, validation, and merging for CLEF platform.
Implements YAML + Pydantic validation strategy from refactor planning.

Author: Raymond Dunn
Version: 2.1.0
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, ConfigDict
from copy import deepcopy

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Configuration Models
# ============================================================================

class DeviceProperties(BaseModel):
    """Device-specific properties (e.g., camera exposure, laser power)."""
    model_config = ConfigDict(extra="allow")


class DeviceConfig(BaseModel):
    """Configuration for a single hardware device."""
    device_name: str = Field(..., description="Micro-Manager device name")
    device_type: str = Field(..., description="Device type (camera, stage, laser, etc.)")
    properties: DeviceProperties = Field(default_factory=DeviceProperties)
    configs: Dict[str, str] = Field(default_factory=dict, description="Micro-Manager config group presets")
    model_config = ConfigDict(extra='allow')


class ShutterConfig(BaseModel):
    """Shutter configuration."""
    device_name: str = Field(..., description="Shutter device name")
    state: bool = Field(..., description="Shutter open state (True=open, False=closed)")


class SystemProperties(BaseModel):
    """System-level Micro-Manager properties."""
    system_name: Optional[str] = Field(None, description="System/microscope name")
    auto_shutter: Optional[bool] = Field(None, description="Auto shutter enabled")
    circular_buffer_mb: Optional[int] = Field(None, gt=0, description="Circular buffer size in MB")
    configs: Dict[str, str] = Field(default_factory=dict, description="System-level config presets")
    shutters: List[ShutterConfig] = Field(default_factory=list, description="Shutter states to set")
    
    model_config = ConfigDict(extra="allow")


class IlluminationChannel(BaseModel):
    """Configuration for an illumination channel."""
    name: str = Field(..., description="Channel name (e.g., '488nm', 'Brightfield')")
    device: str = Field(..., description="Device controlling this channel")
    wavelength: Optional[int] = Field(None, description="Wavelength in nm")
    power: Optional[float] = Field(None, description="Power setting (device-specific units)")
    exposure_ms: Optional[float] = Field(None, description="Exposure time in milliseconds")


class SystemDevices(BaseModel):
    """System devices configuration."""
    camera: Optional[DeviceConfig] = Field(None, description="Camera device configuration")
    stage: Optional[DeviceConfig] = Field(None, description="Stage device configuration")
    illumination_channels: List[IlluminationChannel] = Field(default_factory=list, description="Illumination channels")
    
    model_config = ConfigDict(extra="allow")


class AcquisitionConfig(BaseModel):
    """Acquisition parameters."""
    num_samples: int = Field(100, gt=0, description="Number of frames to acquire")
    model_config = ConfigDict(extra="allow")


class TreatmentDetails(BaseModel):
    """Treatment/condition details."""
    condition: Optional[str] = Field(None, description="Experimental condition")
    model_config = ConfigDict(extra="allow")


class Orientation(BaseModel):
    """Anatomical orientation (domain-specific)."""
    model_config = ConfigDict(extra="allow")


class SubjectDetails(BaseModel):
    """Subject details including treatment and orientation."""
    treatment: Optional[str] = Field(None, description="Experimental treatment (summary)")
    treatment_details: TreatmentDetails = Field(default_factory=TreatmentDetails)
    orientation: Optional[Orientation] = Field(None, description="Anatomical orientation")
    model_config = ConfigDict(extra="allow")


class SubjectMetadata(BaseModel):
    """Experimental subject metadata (generic, not domain-specific)."""
    subject_id: Optional[str] = Field(None, description="Subject identifier")
    subject_type: Optional[str] = Field(None, description="Subject type (e.g., 'C. elegans', 'cell culture')")
    subject_details: SubjectDetails = Field(default_factory=SubjectDetails)
    notes: Optional[str] = Field(None, description="Additional notes")
    model_config = ConfigDict(extra="allow")

    @property
    def genotype(self) -> Optional[str]:
        """Alias for subject_type."""
        return self.subject_type


class ExperimentConfig(BaseModel):
    """Experiment configuration (experiment.yaml)."""
    experiment_name: str = Field(..., description="Experiment name/identifier")
    experimenter: str = Field("Unknown", description="Experimenter name")
    output_dir: str = Field("./data", description="Output directory for data")
    save_images: bool = Field(True, description="Save acquired images")
    save_metadata: bool = Field(True, description="Save metadata JSON")
    save_sample_video: bool = Field(False, description="Save sample/summary video")

    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    subject: SubjectMetadata = Field(default_factory=SubjectMetadata)
    dev_options: Optional[Dict[str, Any]] = Field(None, description="Development options")
    
    model_config = ConfigDict(extra="allow")


class GUIParameters(BaseModel):
    """GUI-specific parameters."""
    enable_gui: bool = Field(False, description="Enable algorithm GUI")
    gui_mode: str = Field("none", description="GUI mode (neural_imaging/behavior/none)")
    save_algorithm_plot: bool = Field(False, description="Save algorithm output plots")

    model_config = ConfigDict(extra="allow")


class StimulusParameters(BaseModel):
    """Stimulus-specific parameters."""
    enabled: bool = Field(False, description="Enable stimulus delivery")

    model_config = ConfigDict(extra="allow")


class AlgorithmParameters(BaseModel):
    """Algorithm-specific parameters."""
    model_config = ConfigDict(extra="allow")


class AlgorithmConfig(BaseModel):
    """Algorithm configuration (algorithm.yaml)."""
    algorithm_type: str = Field("dummy", description="Algorithm class name")
    algorithm_params: AlgorithmParameters = Field(default_factory=AlgorithmParameters)
    stimulus_params: StimulusParameters = Field(default_factory=StimulusParameters)
    gui_params: GUIParameters = Field(default_factory=GUIParameters)

    model_config = ConfigDict(extra="allow")


class StimulusDeviceConfig(BaseModel):
    """Configuration for a specific stimulus device type."""
    type: str = Field(..., description="Stimulus type: widefield_laser, polygon, led, dummy")
    voltage_device: Optional[str] = Field(None, description="Voltage control device")
    voltage_property: Optional[str] = Field(None, description="Voltage property name")
    ttl_device: Optional[str] = Field(None, description="TTL control device")
    ttl_line: Optional[str] = Field(None, description="TTL line name")
    max_volts: Optional[float] = Field(None, description="Maximum voltage")
    intensity_device: Optional[str] = Field(None, description="Intensity control device")
    intensity_property: Optional[str] = Field(None, description="Intensity property name")
    shutter_device: Optional[str] = Field(None, description="Shutter device name")
    slm_device: Optional[str] = Field(None, description="SLM device")
    
    model_config = ConfigDict(extra="allow")


class BackendConfiguration(BaseModel):
    """Backend configuration container."""
    backend_name: str = Field("dummy", description="Backend: pycromanager, pymmcore, or dummy")
    
    model_config = ConfigDict(extra="allow")


class StimulusConfiguration(BaseModel):
    """Stimulus configuration container."""
    stim_interface: str = Field("dummy", description="Stimulus interface class name")
    
    model_config = ConfigDict(extra="allow")


class HardwareConfig(BaseModel):
    """Hardware configuration (hardware.yaml)."""
    backend_configuration: BackendConfiguration = Field(default_factory=BackendConfiguration)
    stimulus_configuration: StimulusConfiguration = Field(default_factory=StimulusConfiguration)
    stimulus_devices: Dict[str, StimulusDeviceConfig] = Field(
        default_factory=dict,
        description="Stimulus device configurations"
    )
    system_devices: SystemDevices = Field(default_factory=SystemDevices)
    system_properties: SystemProperties = Field(default_factory=SystemProperties)
    
    model_config = ConfigDict(extra="allow")
    
    @property
    def backend(self) -> str:
        """Alias for backend_name for backward compatibility."""
        return self.backend_configuration.backend_name
    
    @backend.setter
    def backend(self, value: str) -> None:
        self.backend_configuration.backend_name = value
    
    @property
    def stim_interface(self) -> str:
        """Alias for stim_interface for backward compatibility."""
        return self.stimulus_configuration.stim_interface
    
    @property
    def system_name(self) -> Optional[str]:
        """Alias for system_name for backward compatibility."""
        return self.system_properties.system_name

    @property
    def microscope_name(self) -> Optional[str]:
        """Alias for system_name (legacy name)."""
        return self.system_properties.system_name

    @microscope_name.setter
    def microscope_name(self, value: Optional[str]) -> None:
        self.system_properties.system_name = value
    
    @model_validator(mode='before')
    @classmethod
    def promote_flat_kwargs(cls, data):
        """Promote flat 'backend' and 'stim_interface' kwargs into nested config objects."""
        if isinstance(data, dict):
            if 'backend' in data:
                data = dict(data)
                backend_val = data.pop('backend')
                existing = data.get('backend_configuration', {})
                if isinstance(existing, dict):
                    existing = dict(existing)
                    existing['backend_name'] = backend_val
                    data['backend_configuration'] = existing
                else:
                    data['backend_configuration'] = {'backend_name': backend_val}
            if 'stim_interface' in data and 'stimulus_configuration' not in data:
                data = dict(data)
                data['stimulus_configuration'] = {'stim_interface': data.pop('stim_interface')}
            elif 'stim_interface' in data and 'stimulus_configuration' in data:
                data = dict(data)
                data.pop('stim_interface')  # already nested
            if 'microscope_name' in data:
                data = dict(data)
                name_val = data.pop('microscope_name')
                existing = data.get('system_properties', {})
                if isinstance(existing, dict):
                    existing = dict(existing)
                else:
                    existing = {}
                existing['system_name'] = name_val
                data['system_properties'] = existing
        return data

    @field_validator('backend_configuration')
    @classmethod
    def validate_backend(cls, v):
        allowed = ['pycromanager', 'pymmcore', 'dummy', 'test', 'lorenz_demo', 'ring_attractor_demo', 'screenshot']
        if v.backend_name not in allowed:
            raise ValueError(f"Backend must be one of {allowed}, got '{v.backend_name}'")
        return v
    
    def get_stimulus_device_config(self, interface_name: Optional[str] = None) -> Optional[StimulusDeviceConfig]:
        """Get stimulus device configuration for a given interface."""
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
        """Initialize ConfigManager."""
        if package_root is None:
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
        """Deep merge two config dictionaries (override wins)."""
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
        """Load hardware configuration."""
        default_path = self.defaults_dir / "hardware_default.yaml"
        default_data = self.load_yaml(default_path) if default_path.exists() else {}
        
        if user_path:
            user_data = self.load_yaml(Path(user_path))
            config_data = self.merge_configs(default_data, user_data)
        else:
            config_data = default_data
        
        try:
            self.hardware_config = HardwareConfig(**config_data)
            logger.info(f"Hardware config loaded: backend={self.hardware_config.backend}, "
                       f"stim={self.hardware_config.stim_interface}")
            return self.hardware_config
        except ValidationError as e:
            logger.error(f"Hardware config validation failed: {e}")
            raise
    
    def load_experiment_config(self, user_path: Optional[Path] = None) -> ExperimentConfig:
        """Load experiment configuration."""
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
        """Load algorithm configuration."""
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
        """Load all three configuration files."""
        self.load_hardware_config(hardware_path)
        self.load_experiment_config(experiment_path)
        self.load_algorithm_config(algorithm_path)
        
        logger.info("All configurations loaded successfully")
    
    def validate_config(self) -> bool:
        """Validate loaded configurations for compatibility."""
        if not all([self.hardware_config, self.experiment_config, self.algorithm_config]):
            logger.error("Not all configs loaded - call load_all_configs() first")
            return False
        
        # # Check z-stack configuration consistency
        # if self.experiment_config.acquisition.z_stack:
        #     if self.experiment_config.acquisition.z_planes <= 1:
        #         logger.warning("z_stack enabled but z_planes <= 1")
        #     if self.experiment_config.acquisition.z_step <= 0:
        #         logger.error("z_stack enabled but z_step <= 0")
        #         return False
        
        # Check stimulus parameters consistency
        if self.algorithm_config.stimulus_params.enabled:
            logger.info("Stimulus enabled in algorithm configuration")
        
        logger.info("Configuration validation passed")
        return True
