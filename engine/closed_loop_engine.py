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
            
        # Extract commonly used parameters
        # self._extract_parameters()

    # def _build_legacy_args(self) -> dict[str, Any]:
    #     """
    #     Build legacy args dictionary for components that haven't been refactored yet.
        
    #     This temporary method converts Config objects back to the old flat dict format.
    #     As components are refactored to accept Config objects, calls to this will be removed.
        
    #     Returns:
    #         Dictionary matching old gooey_args format
    #     """
    #     exp = self.experiment_config
    #     hw = self.hardware_config
    #     alg = self.algorithm_config
        
    #     # Build gooey_args format
    #     gooey_args = {
    #         # From ExperimentConfig
    #         "output_folder": exp.output_dir,
    #         "total_frames": exp.acquisition.num_samples,
    #         # "zsize": exp.acquisition.z_planes,
    #         "save_mip": exp.save_sample_video,
    #         "strobe_acquisition": hw.strobe_acquisition,
    #         "strobe_inter_frame_interval": hw.strobe_inter_frame_interval_ms,
    #         # "save_structural_scan": exp.acquisition.save_structural_scan,
    #         # "rec_baseline": exp.acquisition.baseline_samples,
    #         # "z_step_size": exp.z_step_size_um,
            
    #         # Subject metadata
    #         "subject_strain": exp.subject.genotype or "unknown",
    #         "subject_condition": exp.subject.treatment_details.condition,
    #         "atr_concentration": exp.subject.treatment_details.atr_concentration_uM or 0.0,
    #         "nose_orientation": exp.subject.orientation.nose,
    #         "vnc_orientation": exp.subject.orientation.vnc,
    #         "num_eggs": exp.subject.num_eggs,
    #         "experimental_notes": exp.subject.notes or "",
            
    #         # From HardwareConfig
    #         "acquisition_backend": hw.backend,
    #         "mm_configuration_file": hw.mm_config_path or "",
    #         "stim_interface": hw.stim_interface,
    #         "use_static_stim_roi": hw.use_static_stim_roi,
            
    #         # From AlgorithmConfig
    #         "trigger_algorithm": alg.algorithm_type,
    #         "GUI_mode": alg.gui_mode,
    #         "save_alg_model_plot": alg.save_algorithm_plot,
            
    #         # Stimulus parameters
    #         "frames_to_stimulate_for_options": alg.stimulus_params.duration_frames_options,
    #         "stim_intensity_options": alg.stimulus_params.intensity_percent_options,
    #         "stimulus_diameter": alg.algorithm_params.stimulus_diameter_pixels,
            
    #         # Dev options
    #         "input_recording": exp.input_recording_path,
    #         "no_save_images": not exp.save_images,
    #         "no_save_metadata": not exp.save_metadata,
    #         "prefill_wb_ops": exp.dev_options.prefill_wb_ops,
    #         "send_sms": exp.dev_options.send_sms_on_completion,
            
    #         # Additional algorithm params
    #         # "stim_threshold_pos": alg.algorithm_params.stim_threshold_pos,
    #         # "stim_threshold_neg": alg.algorithm_params.stim_threshold_neg,
    #         # "stim_cooldown": alg.algorithm_params.stim_cooldown_frames,
    #         # "skip_stimulation_probability": alg.algorithm_params.skip_stimulation_probability,
    #         # "delay_stimulation_probability": alg.algorithm_params.delay_stimulation_probability,
    #         # "stim_delay_frames_options": alg.algorithm_params.stim_delay_frames_options,
    #         # "stim_onset_list_options": alg.algorithm_params.stim_onset_list,
            
    #         # Backward compatibility - will be removed
    #         "microscope_name": hw.microscope_name or "unknown",
    #     }
        
    #     # Wrap in expected structure
    #     args = {
    #         "gooey_args": gooey_args,
    #         "configs": {},  # Empty for now
    #     }
        
    #     return args
        
    # def _extract_parameters(self):
    #     """Extract frequently used parameters from configs."""
    #     self.zsize = self.gooey_args.get("zsize", 1)
    #     self.samples_to_grab = self.gooey_args.get("total_frames", 100)
    #     self.samples_baseline_window = self.gooey_args.get("rec_baseline", 0)
    #     self.NO_SAVE_DATA = self.gooey_args.get("no_save_images", False)
    #     self.NO_SAVE_METADATA = self.gooey_args.get("no_save_metadata", False)
    #     self.save_mip_movie = self.gooey_args.get("save_mip", False)
    #     self.save_alg_model_plot = self.gooey_args.get("save_alg_model_plot", False)
    #     self.strobe_acquisition = self.hardware_config.strobe_acquisition
    #     self.strobe_inter_frame_interval = self.hardware_config.strobe_inter_frame_interval_ms
    #     self.config_file = self.hardware_config.mm_config_path or ""
    #     self.is_demo_acquisition = self.config_file.endswith("MMConfig_demo.cfg")
    #     self.save_structural_scan = self.gooey_args.get("save_structural_scan", "")
    #     self.prefill_wb_ops = self.gooey_args.get("prefill_wb_ops", False)
    #     self.notify_sms_on_done = self.gooey_args.get("send_sms", True)
    #     self.trigger_alg = self.gooey_args.get("trigger_algorithm", "DummyAlg")
    #     self.acquisition_backend = self.hardware_config.backend
    #     self.gui_mode = self.gooey_args.get("GUI_mode", "neural_imaging")
          
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
        
        # Update args dict with session info (needed by legacy components)
        # self.args["id"] = self.session_id
        # self.args["saveroot"] = self.saveroot
        
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
        # TODO this should reall just be "continuous" vs "discrete"... 
        if self.hardware_config.strobe_acquisition:
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
            if dev_options and getattr(dev_options, 'send_sms_on_completion', False):
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


# def launch_wblive_from_gooey(ops: dict[str, Any] | None = None):
#     """
#     Legacy entry point for launching from Gooey GUI.
    
#     Converts flat gooey_args dict to Config objects and calls new run() method.
    
#     Args:
#         ops: Configuration dictionary from GUI (gooey_args)
#     """
#     if not ops:
#         logger.error("No configuration provided")
#         return
    
#     # Check for dry run mode
#     if ops.get("input_recording") is not None:
#         fname = ops["input_recording"]
#         logger.debug(f"Simulating recording from file: {fname}")
    
#     try:
#         # Convert gooey_args to Config objects
#         configs = convert_gooey_args_to_configs(ops)
        
#         # Run acquisition with new signature
#         engine = ClosedLoopEngine(
#             hardware_config=configs["hardware"],
#             experiment_config=configs["experiment"],
#             algorithm_config=configs["algorithm"]
#         )
#         engine.run()
        
#     except Exception as err:
#         logger.exception(f"Error running acquisition: {err}")
#         raise
#     finally:
#         logger.info("Session complete")
#         sys.exit()


# def convert_gooey_args_to_configs(gooey_args: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Convert legacy gooey_args dictionary to Config objects.
    
#     This function bridges the old flat dict format to the new structured Config objects.
    
#     Args:
#         gooey_args: Dictionary from Gooey GUI with flat key-value pairs
        
#     Returns:
#         Dictionary with keys: "hardware", "experiment", "algorithm" containing Config objects
#     """
#     from config.config_manager import (
#         HardwareConfig,
#         ExperimentConfig,
#         AlgorithmConfig,
#         AcquisitionConfig,
#         SubjectMetadata,
#         TreatmentDetails,
#         Orientation,
#         DevOptions,
#         AlgorithmParameters,
#         StimulusParameters,
#     )
    
#     # Build HardwareConfig
#     hardware_config = HardwareConfig(
#         backend=gooey_args.get("acquisition_backend", "dummy"),
#         mm_config_path=gooey_args.get("mm_configuration_file"),
#         stim_interface=gooey_args.get("stim_interface", "dummy"),
#         microscope_name=gooey_args.get("microscope_name"),
#         strobe_acquisition=gooey_args.get("strobe_acquisition", False),
#         strobe_inter_frame_interval_ms=gooey_args.get("strobe_inter_frame_interval", 80),
#         use_static_stim_roi=gooey_args.get("use_static_stim_roi", False),
#     )
    
#     # Build ExperimentConfig
#     acquisition_config = AcquisitionConfig(
#         num_samples=gooey_args.get("total_frames", 100),
#         z_planes=gooey_args.get("zsize", 1),
#         z_step=gooey_args.get("z_step_size", 1.0),
#         baseline_samples=gooey_args.get("rec_baseline", 0),
#         save_structural_scan=gooey_args.get("save_structural_scan", "none"),
#     )
    
#     treatment_details = TreatmentDetails(
#         condition=gooey_args.get("subject_condition", ""),
#         atr_concentration_uM=gooey_args.get("atr_concentration"),
#     )
    
#     orientation = Orientation(
#         nose=gooey_args.get("nose_orientation"),
#         vnc=gooey_args.get("vnc_orientation"),
#     )
    
#     subject_metadata = SubjectMetadata(
#         genotype=gooey_args.get("subject_strain"),
#         treatment_details=treatment_details,
#         orientation=orientation,
#         num_eggs=gooey_args.get("num_eggs", 0),
#         notes=gooey_args.get("experimental_notes"),
#     )
    
#     dev_options = DevOptions(
#         prefill_wb_ops=gooey_args.get("prefill_wb_ops", False),
#         send_sms_on_completion=gooey_args.get("send_sms", False),
#     )
    
#     experiment_config = ExperimentConfig(
#         experiment_name=gooey_args.get("experiment_name", "default_experiment"),
#         output_dir=gooey_args.get("output_folder", "./data"),
#         save_images=not gooey_args.get("no_save_images", False),
#         save_metadata=not gooey_args.get("no_save_metadata", False),
#         save_sample_video=gooey_args.get("save_sample", False),
#         acquisition=acquisition_config,
#         subject=subject_metadata,
#         z_step_size_um=gooey_args.get("z_step_size", 1.0),
#         input_recording_path=gooey_args.get("input_recording"),
#         dev_options=dev_options,
#     )
    
#     # Build AlgorithmConfig
#     algorithm_params = AlgorithmParameters(
#         stim_threshold_pos=gooey_args.get("stim_threshold_pos", 0.06),
#         stim_threshold_neg=gooey_args.get("stim_threshold_neg", 0.06),
#         stim_cooldown_frames=gooey_args.get("stim_cooldown", 900),
#         skip_stimulation_probability=gooey_args.get("skip_stimulation_probability", 0.1),
#         delay_stimulation_probability=gooey_args.get("delay_stimulation_probability", 0.4),
#         stim_delay_frames_options=gooey_args.get("stim_delay_frames_options", [200, 400]),
#         stim_onset_list=gooey_args.get("stim_onset_list_options", []),
#         stimulus_diameter_pixels=gooey_args.get("stimulus_diameter", 10),
#     )
    
#     stimulus_params = StimulusParameters(
#         duration_frames_options=gooey_args.get("frames_to_stimulate_for_options", [48]),
#         intensity_percent_options=gooey_args.get("stim_intensity_options", [10]),
#     )
    
#     algorithm_config = AlgorithmConfig(
#         algorithm_type=gooey_args.get("trigger_algorithm", "dummy"),
#         gui_mode=gooey_args.get("GUI_mode", "neural_imaging"),
#         save_algorithm_plot=gooey_args.get("save_alg_model_plot", False),
#         algorithm_params=algorithm_params,
#         stimulus_params=stimulus_params,
#     )
    
#     return {
#         "hardware": hardware_config,
#         "experiment": experiment_config,
#         "algorithm": algorithm_config,
#     }


# def run_acquisition(args: dict[str, Any]):
#     """
#     Legacy wrapper for backwards compatibility.
    
#     Args:
#         args: Dictionary containing 'gooey_args' key with configuration
#     """
#     # gooey_args = args.get("gooey_args", {})
#     # configs = convert_gooey_args_to_configs(gooey_args)
#     engine = ClosedLoopEngine(
#         hardware_config=configs["hardware"],
#         experiment_config=configs["experiment"],
#         algorithm_config=configs["algorithm"]
#     )
#     engine.run()


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
        DevOptions,
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

    dev_options = DevOptions()

    experiment_config = ExperimentConfig(
        experiment_name="test_experiment",
        output_dir="./test_output",
        save_images=False,
        save_metadata=False,
        acquisition=acquisition_config,
        subject=subject_metadata,
        dev_options=dev_options,
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