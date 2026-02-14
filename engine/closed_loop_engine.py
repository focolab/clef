"""
Closed-loop acquisition engine for microscopy with real-time stimulus control.

This module provides a class-based wrapper around the acquisition loop,
separating concerns and making the codebase more testable and maintainable.

"""

import sys
import time
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np

# Custom libraries and utils
from utils import wbliveUtils
from utils import MMSubroutines

# Import config models
from config.config_manager import (
    HardwareConfig,
    ExperimentConfig,
    AlgorithmConfig,
)

# Import factory
from algorithms import create_algorithm
from hardware.hardware_manager import HardwareManager
from hardware.stimulus_controllers import create_stimulus_controller
from hardware.data_interface import DataInterface

logger = logging.getLogger(__name__)


class ClosedLoopEngine:
    """
    Main engine for closed-loop microscopy acquisition.
    
    Manages the full lifecycle of an acquisition session including:
    - Hardware initialization (microscope, camera)
    - Algorithm setup (trigger detection)
    - Stimulus interface (via stimulus controllers)
    - Real-time acquisition loop
    - Data saving and cleanup
    """
    
    def __init__(
        self,
        hardware_config: HardwareConfig,
        experiment_config: ExperimentConfig,
        algorithm_config: AlgorithmConfig
    ):
        """
        Initialize the closed-loop engine with configuration objects.
        
        Args:
            hardware_config: Hardware configuration (backend, devices, etc.)
            experiment_config: Experiment configuration (acquisition params, metadata)
            algorithm_config: Algorithm configuration (trigger logic, stimulus params)
        """
        self.hardware_config = hardware_config
        self.experiment_config = experiment_config
        self.algorithm_config = algorithm_config
        
        # Build legacy args structure for components that still expect it
        # This will be gradually eliminated as we refactor components
        # self.args = self._build_legacy_args()
        # self.gooey_args = self.args['gooey_args']
        
        # Execution state - RENAMED FOR GENERIC DATA
        self.is_running = False
        self.sample_count = 0  # RENAMED: was img_count
        self.cooldown_counter = 0
        
        # Components (initialized later)
        self.hardware: HardwareManager = None
        self.mmc = None  # Legacy - will be removed
        # self.xsize = 200  # default for unit tests
        # self.ysize = 200  # default for unit tests
        self.alg = None
        self.stim_controller = None  # stimulus controller loaded later

        # Data storage - RENAMED FOR GENERIC DATA
        # self.samples: np.ndarray | None = None  # RENAMED: was frames
        # self.sample_time_list: list[float] = []  # RENAMED: was frame_time_list
        self.samples_to_grab = experiment_config.acquisition.num_samples
        self.data_interface: DataInterface = None
        self.sample_shape: tuple = None  # NEW: track sample dimensions
        self.sample_dtype: np.dtype = None  # NEW: track sample data type
        self.samples = None
        
        # Timing
        self.next_call: float | None = None

        # initialize runtime experiment config settings
        self.t0 = 1
        self.exp_id = "11111111-11-11-11"  # default for unit tests
        self.roi = (0, 0, 200, 200) # default for unit tests
        
        # Paths and metadata, null initialized for testing
        self.savedir: str | None = None
        self.saveroot: str | None = None
        self.session_id: str | None = None
            
    def initialize_hardware(self):
        """
        Initialize microscope and camera hardware through HardwareManager.
        
        Sets up hardware abstraction layer and configures various settings.
        """
        logger.info("Initializing hardware...")
        
        # Create HardwareManager
        self.hardware = HardwareManager(self.hardware_config)
        
        # Initialize with input recording if provided
        input_recording = getattr(self.experiment_config, 'input_recording_path', None)
        if input_recording:
            logger.info(f"Using input recording: {input_recording}")
            self.hardware.initialize(input_file=input_recording)
        else:
            self.hardware.initialize()

        # Get sample shape and dtype from data interface
        self.data_interface = self.hardware.data
        self.sample_shape = self.data_interface.get_sample_shape()
        self.sample_dtype = self.data_interface.get_sample_dtype()
        # self.samples = self.data_interface.samples # use mutable structure for pointer ref
        
        # TODO Camera configuration is done elsewhere currently but should be done here?
        # This depends on when we want to initialize storage buffer. Currently it's done at "prepare_acquisition"
        
        # For legacy components that still need mmc directly
        # This will be removed as components are refactored
        self.mmc = self.hardware.get_mmc()
        
        logger.info(f"Hardware initialized with shape/dtype: {self.sample_shape}/{self.sample_dtype}")
        
    def initialize_algorithm(self):
        """
        Initialize the closed-loop trigger algorithm using the factory pattern.
        
        Uses AlgorithmFactory to instantiate the correct algorithm based on configuration.
        The factory handles all imports and provides helpful error messages if the
        algorithm type is not found.
        """
        logger.info(f"Initializing algorithm: {self.algorithm_config.algorithm_type}")
        
        try:

            # Create algorithm using factory
            self.alg = create_algorithm(
                algorithm_config=self.algorithm_config,
                experiment_config=self.experiment_config,
                hardware_manager=self.hardware, 
                # local_handles={"mmc": self.mmc} # now optionally in hardware_manager
            )
            
            # Initialize the algorithm's internal model
            self.alg.initialize_model()
            
        except ValueError as err:
            # Algorithm not found in registry
            logger.error(f"Algorithm not found: {err}")
            raise
            
        except Exception as err:
            logger.exception(f"Error initializing algorithm: {err}")
            raise
            
        logger.info("Algorithm initialized successfully")

        
    def initialize_stimulus(self):
        """
        Initialize the stimulus controller using config-based architecture.
        
        NEW: Uses create_stimulus_controller() factory instead of legacy
        StimBaseClass.initialize_stim_interface(). The controller handles
        timing and coordination while hardware_manager handles low-level control.
        """
        logger.info("Initializing stimulus controller...")
        
        try:
            # Get stimulus interface type from hardware config
            stim_interface = self.hardware_config.stim_interface

            # Configure device
            logger.info('Configuring stimulus device')
            self.hardware.stimulus.configure_stimulus(config={'interface_type': stim_interface})
            
            # Create stimulus controller using factory for that device
            self.stim_controller = create_stimulus_controller(
                stim_interface=stim_interface,
                hardware_manager=self.hardware,
                # config=self.args  # Still passing args for backward compatibility
            )

            # Spool controller
            logger.info('Spooling stimulus device')
            self.stim_controller.spool()
            
            logger.info(f"Stimulus controller initialized: {type(self.stim_controller).__name__}")
            
        except Exception as err:
            logger.exception(f"Error initializing stimulus controller: {err}")
            raise
        
    # todo "prepare acquisition" is vague, and it would be better to breakout configuration of individual components.
    # however some have to be done in sequence, like the hardware needs to be configured before the data interface
    # because the data interface takes its dimensions from the hardware
    # however properties like session ID are specified at runtime, not instantiation... not sure what to do here
    def prepare_acquisition(self):
        """
        Prepare for acquisition session.
        
        - Creates output directories
        - Initializes data structures
        - Configures microscope settings
        - Runs structural scans if requested
        """
        logger.info("Preparing acquisition...")
        
        # Create output directory with timestamp
        dt = datetime.today().strftime("%Y%m%d-%H-%M-%S")
        base_savedir = self.experiment_config.output_dir
        self.savedir = os.path.join(base_savedir, dt)
        os.makedirs(self.savedir, exist_ok=True)
        
        self.saveroot = os.path.join(self.savedir, dt)
        self.session_id = dt
        
        # Configure camera for acquisition through hardware manager
        self.data_interface.configure_sampling(self.experiment_config) # send expeirment config to data interface
        self.samples = self.data_interface.samples # use mutable structure for pointer ref
        logger.debug("Data interface prepared for acquisition")
        
        # Run pre-acquisition structural scan if requested
        # self._run_structural_scan_pre()
        
        logger.info(f"Acquisition prepared. Saving to: {self.savedir}")

    # def _run_structural_scan_pre(self):
    #     """Run pre-acquisition structural scan if requested."""
    #     if self.save_structural_scan and "pre" in self.save_structural_scan.lower():
    #         logger.info("Running pre-acquisition structural scan...")
    #         try:
    #             # TODO This still uses MMSubroutines temporarily
    #             # Will be refactored when structural scans are moved to hardware layer
    #             MMSubroutines.run_structural_scan(
    #                 self.save_structural_scan,
    #                 self.mmc,
    #                 self.args,
    #                 self.saveroot,
    #                 self.session_id,
    #                 self.zsize
    #             )
    #         except Exception as err:
    #             logger.error(f"Error in pre-acquisition structural scan: {err}")

    def run_acquisition_loop(self):
        """
        Execute main acquisition loop using data interface.
        
        REFACTORED: Now uses hardware.data.sample_data() instead of
        camera-specific methods. Supports generic data types.
        
        Continuously:
        1. Snap images from camera
        2. Process frames through algorithm
        3. Check for stimulus triggers
        4. Submit stimuli to controller if triggered
        5. Controller manages hardware activation/deactivation
        6. Save data
        
        Raises:
            Exception: When algorithm processing encounters an error
        """
        logger.info(f"Starting acquisition loop for {self.samples_to_grab} samples...")
        
        self.is_running = True
        self.sample_count = 0  # RENAMED: was img_count
        self.cooldown_counter = 0
        self.t0 = time.time()
        
        false_grab_count = 0
        
        # Start acquisition based on mode
        # TODO this should really just be "continuous" vs "discrete"... and should be implemented in the backend/camera not here
        # in the engine 
        if getattr(self.hardware_config, 'strobe_acquisition', False):
            self.next_call = time.time()
            self.data_interface.sample_data()
        else:
            sample_grab_t0 = time.time()
            self.data_interface.stop_sampling()
            self.data_interface.clear_buffer()
            self.data_interface.start_sampling(buffer_size=0)
            
        try:
            while self.is_running and self.sample_count < self.samples_to_grab:
                
                # get data sample
                sample = self.data_interface.sample_data()
                        
                # Record timestamp and store frame
                sample_ndx = self.sample_count
                self.sample_count += 1
                    
                # Periodic logging
                if self.sample_count % 200 == 0:
                    logger.info(f"Sample: {self.sample_count}/{self.samples_to_grab}")
                    
                # Handle cooldown (TODO: move to algorithm/controller)
                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1

                # Process sample through algorithm
                try:
                    self.alg.process_sample(sample, sample_ndx)
                except Exception as err:
                    raise Exception(f"Algorithm error at frame {sample_ndx}: {err}") from err
                    
                # Check for stimulus trigger from algorithm
                stim_params, self.cooldown_counter = self.alg.check_stim(
                    sample_ndx, self.cooldown_counter
                )
                    
                # Submit stimulus params to controller instead of direct stim
                # Controller will manage hardware.stimulus activation/deactivation
                if stim_params:
                    logging.debug(f'Submitting stim params: {stim_params} on image_ndx {sample_ndx}')
                self.stim_controller.submit_stim_params(stim_params, sample_ndx)
                    
                # check controller for stim on next sample (sample_count not sample_ndx)
                self.stim_controller.check_stim(self.sample_count)
                        
        finally:
            self.is_running = False
            t1 = time.time()
            logger.info(f"Acquisition complete. Duration: {t1 - self.t0:.2f}s")

    def save_metadata(self):
        """
        Collect and save metadata from all components.
        
        UPDATED: Gets metadata from stimulus controller instead of direct stim interface.
        
        Aggregates metadata from:
        - Hardware (via HardwareManager)
        - Algorithm
        - Stimulus controller
        - Acquisition settings
        """
        if not self.experiment_config.save_metadata:
            logger.info("Metadata saving disabled")
            return

        logger.info("Saving metadata...")

        # Build metadata dict
        metadata = {}
        metadata["sample_time_list"] = self.data_interface.sample_time_list  # RENAMED
        metadata["t0"] = self.t0
        # metadata["xsize"] = self.data_interface.xsize
        # metadata["ysize"] = self.data_interface.ysize
        metadata["sample_shape"] = self.sample_shape  # NEW
        metadata["sample_dtype"] = str(self.sample_dtype)  # NEW

        # Store configs
        metadata['hardware_config'] = self.hardware_config.model_dump(mode='json')
        metadata['experiment_config'] = self.experiment_config.model_dump(mode='json')
        metadata["algorithm_config"] = self.algorithm_config.model_dump(mode='json')
        
        # Collect hardware metadata through HardwareManager
        if self.hardware and self.hardware.is_initialized:
            metadata["hardware_metadata"] = self.hardware.get_metadata()
        
        # Collect component metadata
        if self.alg:
            metadata["alg_metadata"] = self.alg.get_metadata(args=metadata)
            
        # Get metadata from stimulus controller
        if self.stim_controller:
            metadata["stim_metadata"] = self.stim_controller.get_metadata(args=metadata)
            
        # Save to file
        wbliveUtils.save_metadata(
            savefilename=self.saveroot + "_metadata.json",
            metadata=metadata
        )
        
        # Prefill wb_ops if requested
        # if self.prefill_wb_ops:
        #     wbliveUtils.prefill_wb_ops(savefileroot=self.savedir, metadata=metadata)
            
        logger.info("Metadata saved")

        return metadata
        
    def _save_data(self):
        """
        Save acquired data using data interface.
        
        UPDATED: Delegates to data interface's save_data() method,
        which handles format-specific saving (TIFF, HDF5, NPY, etc.)
        """
        if not self.experiment_config.save_images:
            logger.info("Data saving disabled")
            return
            
        logger.info("Saving data...")
        
        # Delegate to data interface for format-appropriate saving
        self.data_interface.save_data(
            data=self.samples,
            filepath=self.saveroot + ".tiff",
        )
        
        logger.info("Data saved")
        
    def cleanup(self):
        """
        Clean up resources and close connections.
        
        UPDATED: Closes stimulus controller instead of direct stim interface.
        
        Stops acquisition, closes hardware interfaces, and sends notifications.
        """
        logger.info("Cleaning up...")
        
        # Close algorithm
        if self.alg:
            try:
                self.alg.close()
            except Exception as err:
                logger.warning(f"Error closing algorithm: {err}")
                
        # NEW: Close stimulus controller
        if self.stim_controller:
            try:
                self.stim_controller.close()
            except Exception as err:
                logger.warning(f"Error closing stimulus controller: {err}")
                
        # Close hardware through HardwareManager
        if self.hardware:
            try:
                self.hardware.close()
            except Exception as err:
                logger.warning(f"Error closing hardware: {err}")
                
        # Send completion notification if configured
        try:
            dev_options = self.experiment_config.dev_options if self.experiment_config else None
            sms = dev_options.get('send_sms_on_completion', False) if isinstance(dev_options, dict) else getattr(dev_options, 'send_sms_on_completion', False)
            if dev_options and sms:
                wbliveUtils.notify("Acquisition completed")
        except Exception as err:
            logger.warning(f"Error sending completion notification: {err}")

        logger.info("Cleanup complete")
        
    def run(self):
        """
        Orchestrate the full acquisition workflow.
        
        Executes all phases in order:
        1. Initialize hardware (via HardwareManager)
        2. Prepare acquisition
        3. Initialize algorithm
        4. Initialize stimulus controller (NEW: config-based)
        5. Run acquisition loop (NEW: uses controller)
        6. Save data
        7. Cleanup
        
        This is the main entry point for running an acquisition session.
        """
        try:
            logger.info("="*60)
            logger.info("Starting Closed-Loop Acquisition Engine")
            logger.info("="*60)
            
            # Initialization phase
            self.initialize_hardware()
            self.prepare_acquisition()
            self.initialize_algorithm()
            self.initialize_stimulus()  # NEW: Uses config-based controller
            
            # Acquisition phase
            self.run_acquisition_loop()
            
            # Post-processing phase
            self._save_data()  # UPDATED: uses data interface
            self.save_metadata()
            
        except Exception as err:
            logger.exception(f"Error during acquisition: {err}")
            raise
            
        finally:
            # Always cleanup
            self.cleanup()
            
        logger.info("="*60)
        logger.info("Closed-Loop Acquisition Complete")
        logger.info("="*60)


def create_test_config() -> dict[str, Any]:
    """
    Create test configuration objects for running the engine with dummy objects.
    
    Returns:
        Dictionary with Config objects for testing
    """
    from config.config_manager import (
        HardwareConfig,
        ExperimentConfig,
        AlgorithmConfig,
        AcquisitionConfig,
        SubjectMetadata,
        BackendConfiguration,
        StimulusConfiguration,
        SystemProperties,
        AlgorithmConfiguration,
        StimulusParameters,
    )

    hardware_config = HardwareConfig(
        backend_configuration=BackendConfiguration(backend_name="dummy"),
        stimulus_configuration=StimulusConfiguration(stim_interface="dummy"),
        system_properties=SystemProperties(system_name="test"),
    )

    acquisition_config = AcquisitionConfig(
        num_samples=100,
    )

    subject_metadata = SubjectMetadata(
        notes="Test run with dummy objects",
    )

    experiment_config = ExperimentConfig(
        experiment_name="test_experiment",
        output_dir="./test_output",
        save_images=False,
        save_metadata=False,
        acquisition=acquisition_config,
        subject=subject_metadata,
    )

    stimulus_params = StimulusParameters(
        enabled=False,
    )

    algorithm_config = AlgorithmConfig(
        algorithm_type="dummy",
        algorithm_configuration=AlgorithmConfiguration(
            enable_gui=False,
            stimulus_params=stimulus_params,
        ),
    )
    
    return {
        "hardware": hardware_config,
        "experiment": experiment_config,
        "algorithm": algorithm_config,
    }


def run_test():
    """
    Run a test acquisition using dummy objects.
    
    This function demonstrates how to run the ClosedLoopEngine with
    Config objects for testing without real hardware.
    """
    print("="*60)
    print("Running ClosedLoopEngine Test")
    print("="*60)
    
    # Create test configurations
    configs = create_test_config()
    
    # Create and run engine
    try:
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.run()
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
        return True
    except Exception as err:
        print("\n" + "="*60)
        print(f"Test failed with error: {err}")
        print("="*60)
        logger.exception("Full test error traceback:")
        return False


if __name__ == "__main__":
    # Run the test when this module is executed directly
    
    # Set up logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    success = run_test()
    sys.exit(0 if success else 1)