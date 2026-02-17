"""
Algorithm Factory for CLEF Closed-Loop Microscopy System

This module provides a registry-based factory for creating algorithm instances
from configuration objects. It handles dynamic imports and instantiation of
different trigger algorithms used in closed-loop experiments.
"""

import logging
from typing import Dict, Type, Any

logger = logging.getLogger(__name__)

# Import classes at module level, for patching during testing
try:
    from algorithms.dummy import DummyAlg
except ImportError:
    DummyAlg = None

try:
    from algorithms.brainalyzer import Brainalyzer
except ImportError:
    Brainalyzer = None

try:
    from algorithms.demo import LorenzDemoAlgorithm
except ImportError:
    LorenzDemoAlgorithm = None

try:
    from algorithms.demo import DisplayRGBAlgorithm
except ImportError:
    DisplayRGBAlgorithm = None

try:
    from algorithms.demo import RingAttractorAlgorithm
except ImportError:
    RingAttractorAlgorithm = None
    

class AlgorithmRegistry:
    """Registry for algorithm classes."""
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
        self._initialized = False
    
    def register(self, name: str, algorithm_class: Type):
        """Register an algorithm class with a name."""
        self._registry[name] = algorithm_class
        name_str = getattr(algorithm_class, "__name__", str(algorithm_class))
        logger.debug(f"Registered algorithm: {name} -> {name_str}")
    
    def get(self, name: str) -> Type:
        """Get algorithm class by name."""
        if not self._initialized:
            self._initialize_registry()
        
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Unknown algorithm type: '{name}'. "
                f"Available algorithms: {available}"
            )
        return self._registry[name]
    
    def list_algorithms(self) -> list[str]:
        """List all registered algorithm names."""
        if not self._initialized:
            self._initialize_registry()
        return list(self._registry.keys())
    
    def _initialize_registry(self):
        """Lazy initialization of algorithm registry."""
        if self._initialized:
            return
        
        logger.info("Initializing algorithm registry...")

        try:
            # from algorithms.brainalyzer import Brainalyzer
            self.register("Brainalyzer", Brainalyzer)
            self.register("brainalyzer", Brainalyzer)  # lowercase alias
        except ImportError as e:
            logger.warning(f"Could not import Brainalyzer: {e}")

        try:
            # from algorithms.dummy import DummyAlg
            self.register("dummy", DummyAlg)
            self.register("Dummy algorithm (does nothing)", DummyAlg)
        except ImportError as e:
            logger.error(f"Could not import DummyAlg: {e}")
            raise

        # Register Lorenz demo algorithm
        try:
            self.register("LorenzDemo", LorenzDemoAlgorithm)
            self.register("lorenz_demo", LorenzDemoAlgorithm)
            logger.info("Registered LorenzDemoAlgorithm")
        except ImportError as e:
            logger.warning(f"Could not import LorenzDemoAlgorithm: {e}")

        try:
            self.register("DisplayRGBAlgorithm", DisplayRGBAlgorithm)
            self.register("display_rgb_algorithm", DisplayRGBAlgorithm)
            logger.info("Registered DisplayRGBAlgorithm")
        except ImportError as e:
            logger.warning(f"Could not import DisplayRGBAlgorithm: {e}")

        # Register Ring Attractor demo algorithm
        try:
            self.register("RingAttractorDemo", RingAttractorAlgorithm)
            self.register("ring_attractor_demo", RingAttractorAlgorithm)
            logger.info("Registered RingAttractorAlgorithm")
        except ImportError as e:
            logger.warning(f"Could not import RingAttractorAlgorithm: {e}")

        self._initialized = True
        logger.info(f"Algorithm registry initialized with {len(self._registry)} algorithms")


# Global registry instance
_registry = AlgorithmRegistry()


def create_algorithm(
    algorithm_config: 'AlgorithmConfig',
    experiment_config: 'ExperimentConfig',
    hardware_manager: 'HardwareManager',
    local_handles: Dict[str, Any] = None
) -> Any:
    """
    Create an algorithm instance from configuration objects.
    
    This is the main entry point for the factory. It:
    1. Looks up the algorithm class in the registry
    2. Builds the legacy args dict for backwards compatibility
    3. Instantiates the algorithm with appropriate parameters
    4. Initializes the algorithm's model
    
    Args:
        algorithm_config: Algorithm configuration (type, params, etc.)
        experiment_config: Experiment configuration (for building args)
        hardware_config: Hardware configuration (for building args)
        local_handles: Dictionary of local handles (e.g., {'mmc': mmc_instance})
    
    Returns:
        Initialized algorithm instance
    
    Raises:
        ValueError: If algorithm_type is not registered
        Exception: If algorithm initialization fails
    
    Example:
        >>> alg = create_algorithm(
        ...     algorithm_config=AlgorithmConfig(algorithm_type="Brainalyzer"),
        ...     experiment_config=exp_config,
        ...     hardware_config=hw_config,
        ...     local_handles={"mmc": mmc}
        ... )
    """
    if local_handles is None:
        local_handles = {}

    alg_type = algorithm_config.algorithm_type
    logger.info(f"Creating algorithm: {alg_type}")
    
    # Get algorithm class from registry
    try:
        alg_class = _registry.get(alg_type)
    except ValueError as e:
        logger.error(str(e))
        raise
    
    # Build legacy args for algorithms that still expect it
    # args = _build_legacy_args(algorithm_config, experiment_config, hardware_manager.config)
    
    try:
        # Instantiate algorithm
        # Different algorithms have different signatures:
        # - Most take: (args, local_handles={})
        # - Some take: (args) only
        # alg_instance = alg_class(args, local_handles=local_handles)
        alg_instance = alg_class(algorithm_config=algorithm_config, experiment_config=experiment_config, hardware_manager=hardware_manager, local_handles=local_handles)
        
        logger.info(f"Successfully created algorithm: {alg_type}")
        return alg_instance
        
    except Exception as e:
        logger.exception(f"Error creating algorithm {alg_type}: {e}")
        raise


def list_available_algorithms() -> list[str]:
    """
    List all available algorithm types.
    
    Returns:
        List of registered algorithm names
    """
    return _registry.list_algorithms()


def register_algorithm(name: str, algorithm_class: Type):
    """
    Register a custom algorithm class.
    
    This allows external code to register additional algorithms
    without modifying the factory code.
    
    Args:
        name: Name to register algorithm under
        algorithm_class: Algorithm class to register
    
    Example:
        >>> register_algorithm("MyCustomAlg", MyCustomAlgorithm)
    """
    _registry.register(name, algorithm_class)
    logger.info(f"Registered custom algorithm: {name}")
