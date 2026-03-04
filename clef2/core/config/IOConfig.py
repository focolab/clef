import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, ConfigDict

logger = logging.getLogger(__name__)



class DeviceConfig(BaseModel):  
    """
    Configuration for a single input or output device.

    """
    
    device_name: str
    device_type: str
    device_parameters: Dict[str, Any] = Field(default_factory=dict)

    # Extra fields for extensibility
    model_config = ConfigDict(extra="allow")

class IOConfig(BaseModel):
    """
    Configuration for input/output devices.
    
    Defines parameters for input and output devices used in the experiment, including
    device types, parameters, and enabled status.
    """
    
    # Input devices configuration
    input_devices: List[Dict[str, Any]] = Field(default_factory=DeviceConfig)
    
    # Output devices configuration
    output_devices: List[Dict[str, Any]] = Field(default_factory=DeviceConfig)

    # Extra fields for extensibility
    model_config = ConfigDict(extra="allow")