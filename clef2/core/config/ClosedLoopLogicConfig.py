import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class ClosedLoopLogicConfig(BaseModel):
    """
    Configuration for closed-loop logic/algorithm.

    Defines parameters for the closed-loop logic, including algorithm selection,
    algorithm-specific parameters, IO device parameters, and GUI settings.
    """

    # Algorithm selection
    logic_algorithm: str = ""

    # Algorithm-specific parameters
    logic_parameters: Dict[str, Any] = Field(default_factory=dict)

    # Trigger-related algorithm parameters and device configuration
    io_parameters: Dict[str, Any] = Field(default_factory=dict)

    # GUI settings, if applicable to the logic/algorithm
    gui_parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")
