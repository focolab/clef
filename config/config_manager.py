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
    
    devices: Dict[str, DeviceConfig] = Field(default_factory=dict)
    illumination_channels: List[IlluminationChannel] = Field(default_factory=list)
    
    # Calibration data
    polygon_calibration_path: Optional[str] = Field(None, description="Path to polygon calibration JSON")
    
    model_config = ConfigDict(extra="allow")  # Allow additional hardware-specific fields
    
    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v):
        allowed = ['pycromanager', 'pymmcore', 'dummy']
        if v not in allowed:
            raise ValueError(f"Backend must be one of {allowed}, got '{v}'")
        return v


class AcquisitionConfig(BaseModel):
    """Acquisition parameters."""
    num_frames: int = Field(100, gt=0, description="Number of frames to acquire")
    frame_rate: Optional[float] = Field(None, gt=0, description="Target frame rate (Hz)")
    z_stack: bool = Field(False, description="Enable Z-stack acquisition")
    z_start: float = Field(0.0, description="Z-stack start position (µm)")
    z_end: float = Field(0.0, description="Z-stack end position (µm)")
    z_step: float = Field(1.0, gt=0, description="Z-stack step size (µm)")
    
    @field_validator('z_end')
    @classmethod
    def validate_z_range(cls, v, info):
        values = info.data
        if values.get('z_stack') and v <= values.get('z_start', 0):
            raise ValueError("z_end must be greater than z_start for z-stack acquisition")
        return v


class SubjectMetadata(BaseModel):
    """Experimental subject metadata (generic, not worm-specific)."""
    subject_id: Optional[str] = Field(None, description="Subject identifier")
    subject_type: Optional[str] = Field(None, description="Subject type (e.g., 'C. elegans', 'cell culture')")
    genotype: Optional[str] = Field(None, description="Genetic background or strain")
    treatment: Optional[str] = Field(None, description="Experimental treatment")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    model_config = ConfigDict(extra="allow")  # Allow domain-specific fields


class ExperimentConfig(BaseModel):
    """Experiment configuration (experiment.yaml)."""
    experiment_name: str = Field(..., description="Experiment name/identifier")
    experimenter: str = Field("Unknown", description="Experimenter name")
    output_dir: str = Field("./data", description="Output directory for data")
    save_images: bool = Field(True, description="Save acquired images")
    save_metadata: bool = Field(True, description="Save metadata JSON")
    
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    subject: SubjectMetadata = Field(default_factory=SubjectMetadata)
    
    model_config = ConfigDict(extra="allow")


class AlgorithmParameters(BaseModel):
    """Algorithm-specific parameters."""
    model_config = ConfigDict(extra="allow")

class StimulusParameters(BaseModel):
    """Stimulus-specific parameters."""
    enabled: bool = Field(False, description="Enable stimulus delivery")
    randomize: bool = Field(False, description="Randomize stimulus timing")
    
    model_config = ConfigDict(extra="allow")


class AlgorithmConfig(BaseModel):
    """Algorithm configuration (algorithm.yaml)."""
    algorithm_type: str = Field("dummy", description="Algorithm class name")
    enable_gui: bool = Field(False, description="Enable algorithm GUI")
    
    algorithm_params: AlgorithmParameters = Field(default_factory=AlgorithmParameters)
    stimulus_params: StimulusParameters = Field(default_factory=StimulusParameters)
    
    model_config = ConfigDict(extra="allow")


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
        
        # Example compatibility check: dummy backend should have dummy stim
        if self.hardware_config.backend == "dummy":
            if self.hardware_config.stim_interface != "dummy":
                logger.warning(f"Backend is 'dummy' but stim_interface is "
                             f"'{self.hardware_config.stim_interface}' - may not work")
        
        logger.info("Configuration validation passed")
        return True