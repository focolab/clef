"""
Base class for all closed-loop logic algorithms in CLEF.

Subclasses must set `logic_class` to a unique string that matches
the `logic_class` field in the logic config YAML.  Registration
happens automatically via __init_subclass__.

The base class is fully instantiable with no-op implementations,
acting as a dummy algorithm by default.
"""

import logging
from typing import Any, ClassVar, Dict, Optional, Type

logger = logging.getLogger(__name__)


class BaseClosedLoopLogic:
    """Base class for closed-loop logic. All methods are no-ops by default."""

    # --- registry --------------------------------------------------------
    _registry: ClassVar[Dict[str, Type["BaseClosedLoopLogic"]]] = {}

    logic_class: ClassVar[Optional[str]] = "base_closed_loop_logic"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.logic_class is not None:
            BaseClosedLoopLogic._registry[cls.logic_class] = cls
            logger.debug(f"Registered logic: {cls.logic_class} -> {cls.__name__}")

    @classmethod
    def get_class(cls, logic_class: str) -> Type["BaseClosedLoopLogic"]:
        """Look up a registered logic class by its logic_class key."""
        if logic_class not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown logic class: '{logic_class}'. Available: {available}"
            )
        return cls._registry[logic_class]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return all registered logic_class keys."""
        return list(cls._registry.keys())

    # --- lifecycle -------------------------------------------------------

    def __init__(
        self,
        name: str,
        config: Dict[str, Any] | None = None,
        output_devices: Dict[str, Any] | None = None,
        gui_parameters: Dict[str, Any] | None = None,
    ):
        """
        Args:
            name: Instance name from config (logic_algorithm_name).
            config: The logic_parameters dict from config.
            output_devices: Resolved output device references from IOManager.
            gui_parameters: GUI configuration for this logic instance.
        """
        self.name = name
        self.config = config or {}
        self.output_devices = output_devices or {}
        self.gui_parameters = gui_parameters or {}

    def initialize_model(self):
        """Initialize the algorithm model. Called once before processing begins."""
        pass

    def process_sample(self, sample: Any) -> Dict[str, Any]:
        """
        Process a single data sample.

        Returns:
            Dict with at minimum {"triggered": bool, "trigger_value": float}
        """
        return {"triggered": False, "trigger_value": 0.0}

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the closed-loop logic."""
        return {"name": self.name, "logic_class": self.logic_class}

    def close(self):
        """Clean up any resources used by the closed-loop logic."""
        pass


# __init_subclass__ doesn't fire for the class it's defined on,
# so register the base class manually.
BaseClosedLoopLogic._registry[BaseClosedLoopLogic.logic_class] = BaseClosedLoopLogic
