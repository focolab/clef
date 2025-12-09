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
# TODO clean this
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
        
        # Import and register algorithms
        # Register DummyAlg if available
        # if DummyAlg is not None:
        #     self.register("dummy", DummyAlg)
        #     self.register("Dummy algorithm (does nothing)", DummyAlg)
        
        # # Register Brainalyzer if available
        # if Brainalyzer is not None:
        #     self.register("Brainalyzer", Brainalyzer)
        #     self.register("brainalyzer", Brainalyzer)

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


        # try:
        #     from lib import DynamicRangeDeriv
        #     self.register("Dynamic range deriv", DynamicRangeDeriv.DynamicRangeDeriv)
        # except ImportError as e:
        #     logger.warning(f"Could not import DynamicRangeDeriv: {e}")
        
        # try:
        #     from lib import RoiDeriv
        #     self.register("RoiDeriv", RoiDeriv.RoiDeriv)
        # except ImportError as e:
        #     logger.warning(f"Could not import RoiDeriv: {e}")
        
        # try:
        #     from lib import StimOnsetFromList
        #     self.register("StimOnsetFromList", StimOnsetFromList.StimOnsetFromList)
        # except ImportError as e:
        #     logger.warning(f"Could not import StimOnsetFromList: {e}")
        
        # try:
        #     from lib import PointAndClick
        #     self.register("PointAndClick", PointAndClick.PointAndClick)
        # except ImportError as e:
        #     logger.warning(f"Could not import PointAndClick: {e}")
        
        # try:
        #     from lib import HammerOfDawn
        #     self.register("HammerOfDawn", HammerOfDawn.HammerOfDawn)
        # except ImportError as e:
        #     logger.warning(f"Could not import HammerOfDawn: {e}")
        

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
    args = _build_legacy_args(algorithm_config, experiment_config, hardware_manager.config)
    
    try:
        # Instantiate algorithm
        # Different algorithms have different signatures:
        # - Most take: (args, local_handles={})
        # - Some take: (args) only
        # alg_instance = alg_class(args, local_handles=local_handles)
        alg_instance = alg_class(algorithm_config=algorithm_config, experiment_config=experiment_config, hardware_manager=hardware_manager, args=args, local_handles=local_handles)
        
        logger.info(f"Successfully created algorithm: {alg_type}")
        return alg_instance
        
    except Exception as e:
        logger.exception(f"Error creating algorithm {alg_type}: {e}")
        raise


def _build_legacy_args(
    algorithm_config: 'AlgorithmConfig',
    experiment_config: 'ExperimentConfig',
    hardware_config: 'HardwareConfig'
) -> Dict[str, Any]:
    """
    Build legacy args dictionary for backwards compatibility.
    
    This recreates the flat dict structure that existing algorithms expect.
    As algorithms are refactored to accept Config objects directly,
    this function can be gradually simplified or removed.
    
    Args:
        algorithm_config: Algorithm configuration
        experiment_config: Experiment configuration
        hardware_config: Hardware configuration
    
    Returns:
        Dictionary matching old gooey_args format
    """
    exp = experiment_config
    hw = hardware_config
    alg = algorithm_config
    
    # Build gooey_args format
    gooey_args = {
        # From ExperimentConfig
        "output_folder": exp.output_dir,
        "total_frames": exp.acquisition.num_samples,
        "zsize": exp.acquisition.z_planes,
        "save_mip": exp.save_mip_video,
        "save_structural_scan": exp.acquisition.save_structural_scan,
        "rec_baseline": exp.acquisition.baseline_samples,
        "z_step_size": exp.z_step_size_um,
        
        # Subject metadata
        "subject_strain": exp.subject.genotype or "unknown",
        "subject_condition": exp.subject.treatment_details.condition,
        "atr_concentration": exp.subject.treatment_details.atr_concentration_uM or 0.0,
        "nose_orientation": exp.subject.orientation.nose,
        "vnc_orientation": exp.subject.orientation.vnc,
        "num_eggs": exp.subject.num_eggs,
        "experimental_notes": exp.subject.notes or "",
        
        # From HardwareConfig
        "acquisition_backend": hw.backend,
        "mm_configuration_file": hw.mm_config_path or "",
        "stim_interface": hw.stim_interface,
        "use_static_stim_roi": hw.use_static_stim_roi,
        "strobe_acquisition": hw.strobe_acquisition,
        "strobe_inter_frame_interval": hw.strobe_inter_frame_interval_ms,
        "microscope_name": hw.microscope_name or "unknown",
        
        # From AlgorithmConfig
        "trigger_algorithm": alg.algorithm_type,
        "GUI_mode": alg.gui_mode,
        "save_alg_model_plot": alg.save_algorithm_plot,
        
        # Stimulus parameters
        "frames_to_stimulate_for_options": alg.stimulus_params.duration_frames_options,
        "stim_intensity_options": alg.stimulus_params.intensity_percent_options,
        "stimulus_diameter": alg.algorithm_params.stimulus_diameter_pixels,
        
        # Dev options
        "input_recording": exp.input_recording_path,
        "no_save_images": not exp.save_images,
        "no_save_metadata": not exp.save_metadata,
        "prefill_wb_ops": exp.dev_options.prefill_wb_ops,
        "send_sms": exp.dev_options.send_sms_on_completion,
        
        # Additional algorithm params
        "stim_cooldown": alg.algorithm_params.stim_cooldown_frames,
        "skip_stimulation_probability": alg.algorithm_params.skip_stimulation_probability,
        "delay_stimulation_probability": alg.algorithm_params.delay_stimulation_probability,
        "stim_delay_frames_options": alg.algorithm_params.stim_delay_frames_options,
        "stim_onset_list_options": alg.algorithm_params.stim_onset_list,
    }
    
    # Wrap in expected structure
    args = {
        "id": '11111111-11-11-11',
        'roi': [0, 0, 200, 200],
        "gooey_args": gooey_args,
        "configs": {
            "algorithm": algorithm_config,
            "experiment": experiment_config,
            "hardware": hardware_config,
        },
    }
    
    return args


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
