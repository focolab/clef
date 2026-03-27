import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, ConfigDict

logger = logging.getLogger(__name__)

class SessionConfig(BaseModel):
    """
    Configuration for experimental session.
    
    Contains metadata about the session, acquisition parameters, and references to logic and IO configs.
    """

    # Session metadata
    session_name: str = "test_experiment"
    user_name: str = "Unknown"
    sample_data_dir: str = "./data"

    # Runtime session ID (set by CLI at launch, format: yyyymmdd-hh-mm-ss)
    session_id: str = ""

    # Acquisition parameters
    session_parameters: Dict[str, Any] = Field(default_factory=dict)

    # References to logic and IO configs
    logic_config_path: Optional[Path] = None
    io_config_path: Optional[Path] = None

    # Extra fields for extensibility
    model_config = ConfigDict(extra="allow")