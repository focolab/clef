import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class LogicAlgorithmEntry(BaseModel):
    """Configuration for a single logic algorithm instance."""

    logic_algorithm_name: Optional[str] = None
    logic_class: str = ""
    logic_parameters: Dict[str, Any] = Field(default_factory=dict)
    io_parameters: Dict[str, Any] = Field(default_factory=dict)
    gui_parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class ClosedLoopLogicConfig(BaseModel):
    """Configuration for all closed-loop logic algorithms."""

    logic_algorithms: List[LogicAlgorithmEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")
