import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class InputDeviceConfig(BaseModel):
    """Configuration for a single input device."""

    input_device_name: Optional[str] = None
    input_device_type: Optional[str] = None
    input_device_class: Optional[str] = None
    data_interface_class: Optional[str] = None
    input_device_parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class OutputDeviceConfig(BaseModel):
    """Configuration for a single output device."""

    output_device_name: Optional[str] = None
    output_device_type: Optional[str] = None
    output_device_class: Optional[str] = None
    output_device_parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class IOConfig(BaseModel):
    """
    Configuration for input/output devices.

    Defines parameters for input and output devices used in the experiment, including
    device types, parameters, and enabled status.
    """

    input_devices: List[InputDeviceConfig] = Field(default_factory=list)
    output_devices: List[OutputDeviceConfig] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")
