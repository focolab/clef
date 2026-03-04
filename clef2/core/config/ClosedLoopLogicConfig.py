import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, ConfigDict

logger = logging.getLogger(__name__)

class ClosedLoopLogicConfig(BaseModel):
    """
    Configuration for closed-loop logic/algorithm.
    
    Defines parameters for the closed-loop logic, including algorithm-specific
    parameters, IO device parameters, and GUI settings.
    """
    
    # Algorithm-specific parameters
    logic_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Trigger-related algorithm parameters
    io_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # GUI settings, if applicable to the logic/algorithm
    gui_parameters: Dict[str, Any] = Field(default_factory=dict)

    # Extra fields for extensibility
    model_config = ConfigDict(extra="allow")