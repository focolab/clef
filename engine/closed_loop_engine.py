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
from lib import wbliveUtils as utils
from lib import MMSubroutines

# Import config models
from config.config_manager import (
    HardwareConfig,
    ExperimentConfig,
    AlgorithmConfig,
)

# Import factory
from algorithms import create_algorithm

# Import hardware manager and stimulus controllers
from hardware.hardware_manager import HardwareManager
from hardware.stimulus_controllers import create_stimulus_controller

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
        self.args = self._build_legacy_args()
        self.gooey_args = self.args['gooey_args']
        
        # Execution state
        self.is_running = False
        self.frame_count = 0
        self.img_count = 0
        self.cooldown_counter = 0
        
        # Components (initialized later)
        self.hardware: HardwareManager = None
        self.mmc = None  # Legacy - will be removed
        self.xsize = 200  # default for unit tests
        self.ysize = 200  # default for unit tests
        self.roi = (0, 0, 200, 200)
        self.alg = None
        self.stim_controller = None  # stimulus controller
        self.t0 = 1.  # default for unit tests
        self.args["t0"] = self.t0
        self.args["id"] = "11111111-11-11-11"  # default for unit tests
        self.args["saveroot"] = 'C:/Users/rldun/Downloads/'
        
        # Data storage
        self.frames: np.ndarray | None = None
        self.frame_time_list: list[float] = []
        
        # Timing
        self.next_call: float | None = None
        
        # Paths and metadata
        self.savedir: str | None = None
        self.saveroot: str | None = None
        self.session_id: str | None = None  
            
        # Extract commonly used parameters
        self._extract_parameters()

    def _build_legacy_args(self) -> dict[str, Any]:
        """
        Build legacy args dictionary for components that haven't been refactored yet.
        
        This temporary method converts Config objects back to the old flat dict format.
        As components are refactored to accept Config objects, calls to this will be removed.
        
        Returns:
            Dictionary matching old gooey_args format
        """
        exp = self.experiment_config
        hw = self.hardware_config
        alg = self.algorithm_config
        
        # Build gooey_args format
        gooey_args = {
            # From ExperimentConfig
            "output_folder": exp.output_dir,
            "total_frames": exp.acquisition.num_frames,
            "zsize": exp.acquisition.z_planes,
            "save_mip": exp.save_mip_video,
            "strobe_acquisition": hw.strobe_acquisition,
            "strobe_inter_frame_interval": hw.strobe_inter_frame_interval_ms,
            "save_structural_scan": exp.acquisition.save_structural_scan,
            "rec_baseline": exp.acquisition.baseline_frames,
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
            "stim_threshold_pos": alg.algorithm_params.stim_threshold_pos,
            "stim_threshold_neg": alg.algorithm_params.stim_threshold_neg,
            "stim_cooldown": alg.algorithm_params.stim_cooldown_frames,
            "skip_stimulation_probability": alg.algorithm_params.skip_stimulation_probability,
            "delay_stimulation_probability": alg.algorithm_params.delay_stimulation_probability,
            "stim_delay_frames_options": alg.algorithm_params.stim_delay_frames_options,
            "stim_onset_list_options": alg.algorithm_params.stim_onset_list,
            
            # Backward compatibility - will be removed
            "microscope_name": hw.microscope_name or "unknown",
        }
        
        # Wrap in expected structure
        args = {
            "gooey_args": gooey_args,
            "configs": {},  # Empty for now
        }
        
        return args
        
    def _extract_parameters(self):
        """Extract frequently used parameters from configs."""
        self.zsize = self.gooey_args.get("zsize", 1)
        self.frames_to_grab = self.gooey_args.get("total_frames", 100)
        self.frames_baseline_window = self.gooey_args.get("rec_baseline", 0)
        self.no_save_images = self.gooey_args.get("no_save_images", False)
        self.no_save_metadata = self.gooey_args.get("no_save_metadata", False)
        self.save_mip_movie = self.gooey_args.get("save_mip", False)
        self.save_alg_model_plot = self.gooey_args.get("save_alg_model_plot", False)
        self.strobe_acquisition = self.hardware_config.strobe_acquisition
        self.strobe_inter_frame_interval = self.hardware_config.strobe_inter_frame_interval_ms
        self.config_file = self.hardware_config.mm_config_path or ""
        self.is_demo_acquisition = self.config_file.endswith("MMConfig_demo.cfg")
        self.save_structural_scan = self.gooey_args.get("save_structural_scan", "")
        self.prefill_wb_ops = self.gooey_args.get("prefill_wb_ops", False)
        self.notify_sms_on_done = self.gooey_args.get("send_sms", True)
        self.trigger_alg = self.gooey_args.get("trigger_algorithm", "DummyAlg")
        self.acquisition_backend = self.hardware_config.backend
        self.gui_mode = self.gooey_args.get("GUI_mode", "neural_imaging")
          
    def initialize_hardware(self):
        """
        Initialize microscope and camera hardware through HardwareManager.
        
        Sets up hardware abstraction layer and configures ROI settings.
        """
        logger.info("Initializing hardware...")
        
        # Create HardwareManager
        self.hardware = HardwareManager(self.hardware_config)
        
        # Initialize with input recording if provided
        input_recording = self.experiment_config.input_recording_path
        if input_recording:
            logger.info(f"Using input recording: {input_recording}")
            self.hardware.initialize(input_file=input_recording)
        else:
            self.hardware.initialize()
        
        # Get ROI from camera interface
        self.roi = self.hardware.camera.get_roi()
        self.args["roi"] = self.roi
        
        # Extract image dimensions
        self.xsize = self.roi[2]
        self.ysize = self.roi[3]
        
        # For legacy components that still need mmc directly
        # This will be removed as components are refactored
        self.mmc = self.hardware.get_mmc()
        
        logger.info(f"Hardware initialized with camera ROI: {self.roi}")
        
    def initialize_algorithm(self):
        """
        Initialize the closed-loop trigger algorithm using the factory pattern.
        
        Uses AlgorithmFactory to instantiate the correct algorithm based on configuration.
        The factory handles all imports and provides helpful error messages if the
        algorithm type is not found.
        """
        logger.info(f"Initializing algorithm: {self.trigger_alg}")
        
        try:

            # Create algorithm using factory
            self.alg = create_algorithm(
                algorithm_config=self.algorithm_config,
                experiment_config=self.experiment_config,
                hardware_config=self.hardware_config,
                local_handles={"mmc": self.mmc}
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
            
            # Create stimulus controller using factory
            self.stim_controller = create_stimulus_controller(
                stim_interface=stim_interface,
                hardware_manager=self.hardware,
                config=self.args  # Still passing args for backward compatibility
            )
            
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
        self.args["id"] = self.session_id
        self.args["saveroot"] = self.saveroot
        
        # Initialize frame storage (requires hardware initialization for dimensions)
        self.frames = np.zeros(
            (self.frames_to_grab, self.ysize, self.xsize), 
            dtype=np.uint16
        )
        self.frame_time_list = []
        
        # Configure camera for acquisition through hardware manager
        self._prepare_camera_acquisition()
        
        # Run pre-acquisition structural scan if requested
        self._run_structural_scan_pre()
        
        logger.info(f"Acquisition prepared. Saving to: {self.savedir}")

    def _prepare_camera_acquisition(self):
        """Configure camera for live acquisition through HardwareManager."""
        camera = self.hardware.camera
        
        # Clear buffer
        camera.clear_buffer()
        
        # Set buffer size based on config
        buffer_size = 10000  # Default, can be made configurable
        
        # Start acquisition will be called in run_acquisition_loop
        logger.debug("Camera prepared for acquisition")

    def _run_structural_scan_pre(self):
        """Run pre-acquisition structural scan if requested."""
        if self.save_structural_scan and "pre" in self.save_structural_scan.lower():
            logger.info("Running pre-acquisition structural scan...")
            try:
                # TODO This still uses MMSubroutines temporarily
                # Will be refactored when structural scans are moved to hardware layer
                MMSubroutines.run_structural_scan(
                    self.save_structural_scan,
                    self.mmc,
                    self.args,
                    self.saveroot,
                    self.session_id,
                    self.zsize
                )
            except Exception as err:
                logger.error(f"Error in pre-acquisition structural scan: {err}")

    def run_acquisition_loop(self):
        """
        Execute the main acquisition loop using HardwareManager.
        
        UPDATED: Now uses stimulus controller to handle stimulus events instead
        of direct stimulus interface calls. The algorithm returns stim_params,
        which are submitted to the controller, which then manages hardware.stimulus.
        
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
        logger.info(f"Starting acquisition loop for {self.frames_to_grab} frames...")
        
        self.is_running = True
        self.img_count = 0
        self.cooldown_counter = 0
        self.t0 = time.time()
        
        false_grab_count = 0
        camera = self.hardware.camera
        
        # Start acquisition based on mode
        if self.strobe_acquisition:
            self.next_call = time.time()
            camera.snap_image()
        else:
            frame_grab_t0 = time.time()
            camera.stop_acquisition()
            camera.clear_buffer()
            camera.start_acquisition(buffer_size=10000)
            
        try:
            while self.is_running and self.img_count < self.frames_to_grab:
                rem = camera.get_remaining_image_count()
                
                while (rem > 0 or self.strobe_acquisition) and self.img_count < self.frames_to_grab:
                    # Grab image from buffer
                    try:
                        if self.strobe_acquisition:
                            img = camera.get_image()
                        else:
                            img = camera.pop_next_image()
                            self.next_call = frame_grab_t0
                    except Exception as err:
                        false_grab_count += 1
                        logger.debug(f"False grab #{false_grab_count}: {err}")
                        continue
                        
                    # Record timestamp and store frame
                    frame_grab_t0 = time.time()
                    self.frame_time_list.append(
                        np.round(frame_grab_t0 - self.t0, decimals=4)
                    )
                    self.frames[self.img_count, :, :] = img
                    image_ndx = self.img_count
                    self.img_count += 1
                    
                    # Periodic logging
                    if self.img_count % 200 == 0:
                        logger.info(f"Frame: {self.img_count}/{self.frames_to_grab}")
                    
                    # Handle cooldown (TODO: move to algorithm/controller)
                    if self.cooldown_counter > 0:
                        self.cooldown_counter -= 1
                        
                    # Process frame through algorithm
                    zndx = image_ndx % self.zsize
                    try:
                        self.alg.process_frame(img, zndx)
                    except Exception as err:
                        raise Exception(f"Algorithm error at frame {image_ndx}, z={zndx}: {err}") from err
                    
                    # Check for stimulus trigger from algorithm
                    stim_params, self.cooldown_counter = self.alg.check_stim(
                        image_ndx, self.cooldown_counter
                    )
                    
                    # Submit stimulus params to controller instead of direct stim
                    # Controller will manage hardware.stimulus activation/deactivation
                    if stim_params:
                        logging.debug(f'Submitting stim params: {stim_params} on image_ndx {image_ndx}')
                        self.stim_controller.submit_stim_params(stim_params, image_ndx)
                    
                    # Volume completion handling
                    # Controller checks if this frame should trigger hardware changes
                    if zndx == self.zsize - 1:
                        self.stim_controller.check_stim(self.img_count)
                        
                    # Handle strobe timing
                    if self.strobe_acquisition:
                        nowtime = time.time()
                        self.next_call = self.next_call + self.strobe_inter_frame_interval / 1000
                        
                        if self.next_call - nowtime < 0:
                            logger.warning(
                                f"Strobe delay exceeded interval! Frame: {image_ndx}"
                            )
                        else:
                            time.sleep(self.next_call - nowtime)
                            
                        camera.snap_image()
                    else:
                        rem = camera.get_remaining_image_count()
                        
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
        if self.no_save_metadata:
            logger.info("Metadata saving disabled")
            return
            
        logger.info("Saving metadata...")
        
        # Build metadata dict
        metadata = dict(self.args)
        metadata["frame_time_list"] = self.frame_time_list
        metadata["t0"] = self.t0
        metadata["xsize"] = self.xsize
        metadata["ysize"] = self.ysize

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
        utils.save_metadata(
            savefilename=self.saveroot + "_metadata.json",
            metadata=metadata
        )
        
        # Prefill wb_ops if requested
        if self.prefill_wb_ops:
            utils.prefill_wb_ops(savefileroot=self.savedir, metadata=metadata)
            
        logger.info("Metadata saved")

        return metadata
        
    def _save_images(self):
        """Save acquired image stack."""
        if self.no_save_images:
            logger.info("Image saving disabled")
            return
            
        logger.info("Saving images...")

        # TODO migrate this it shouldn't be here maybe utils?
        MMSubroutines.saveScanTiffs(
            fname=self.saveroot + ".tiff",
            img_array=self.frames
        )
        logger.info("Images saved")
        
    def _save_visualizations(self, mip_fps: float | None = None):
        """Save algorithm plots and MIP movies."""
        if self.save_alg_model_plot and self.alg:
            logger.info("Saving algorithm model plot...")
            self.alg.plot_model(savefilename=self.saveroot + "_live_stim_fig.svg")
            
        if self.save_mip_movie:
            logger.info("Generating MIP movie...")
            exposure = self.hardware.camera.get_exposure() if self.hardware else mip_fps
            utils.generate_mip_movie(
                savefilename=self.saveroot + "_mip_movie",
                frames=self.frames,
                zsize=self.zsize,
                exposure=exposure,
                GUI_mode=self.gui_mode,
            )
            
    def _post_acquisition_structural_scan(self):
        """Run structural scan after acquisition if requested."""
        if "NeuroPAL" in self.save_structural_scan:
            logger.info("Running post-acquisition structural scan...")
            try:
                MMSubroutines.run_structural_scan(
                    self.save_structural_scan,
                    self.mmc,
                    self.args,
                    self.saveroot,
                    self.session_id,
                    self.zsize
                )
            except Exception as err:
                logger.error(f"Error in post-acquisition structural scan: {err}")
                
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
                
        # Send notification
        if self.notify_sms_on_done:
            try:
                msg = f"Your wb-live recording {self.session_id} has completed."
                utils.notify(msg, interface="twilio-sms")
            except Exception as err:
                logger.warning(f"Error sending notification: {err}")
                
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
            self._post_acquisition_structural_scan()
            self._save_images()
            self._save_visualizations()
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


def launch_wblive_from_gooey(ops: dict[str, Any] | None = None):
    """
    Legacy entry point for launching from Gooey GUI.
    
    Converts flat gooey_args dict to Config objects and calls new run() method.
    
    Args:
        ops: Configuration dictionary from GUI (gooey_args)
    """
    if not ops:
        logger.error("No configuration provided")
        return
    
    # Check for dry run mode
    if ops.get("input_recording") is not None:
        fname = ops["input_recording"]
        logger.debug(f"Simulating recording from file: {fname}")
    
    try:
        # Convert gooey_args to Config objects
        configs = convert_gooey_args_to_configs(ops)
        
        # Run acquisition with new signature
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.run()
        
    except Exception as err:
        logger.exception(f"Error running acquisition: {err}")
        raise
    finally:
        logger.info("Session complete")
        sys.exit()


def convert_gooey_args_to_configs(gooey_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert legacy gooey_args dictionary to Config objects.
    
    This function bridges the old flat dict format to the new structured Config objects.
    
    Args:
        gooey_args: Dictionary from Gooey GUI with flat key-value pairs
        
    Returns:
        Dictionary with keys: "hardware", "experiment", "algorithm" containing Config objects
    """
    from config.config_manager import (
        HardwareConfig,
        ExperimentConfig,
        AlgorithmConfig,
        AcquisitionConfig,
        SubjectMetadata,
        TreatmentDetails,
        Orientation,
        DevOptions,
        AlgorithmParameters,
        StimulusParameters,
    )
    
    # Build HardwareConfig
    hardware_config = HardwareConfig(
        backend=gooey_args.get("acquisition_backend", "dummy"),
        mm_config_path=gooey_args.get("mm_configuration_file"),
        stim_interface=gooey_args.get("stim_interface", "dummy"),
        microscope_name=gooey_args.get("microscope_name"),
        strobe_acquisition=gooey_args.get("strobe_acquisition", False),
        strobe_inter_frame_interval_ms=gooey_args.get("strobe_inter_frame_interval", 80),
        use_static_stim_roi=gooey_args.get("use_static_stim_roi", False),
    )
    
    # Build ExperimentConfig
    acquisition_config = AcquisitionConfig(
        num_frames=gooey_args.get("total_frames", 100),
        z_planes=gooey_args.get("zsize", 1),
        z_step=gooey_args.get("z_step_size", 1.0),
        baseline_frames=gooey_args.get("rec_baseline", 0),
        save_structural_scan=gooey_args.get("save_structural_scan", "none"),
    )
    
    treatment_details = TreatmentDetails(
        condition=gooey_args.get("subject_condition", ""),
        atr_concentration_uM=gooey_args.get("atr_concentration"),
    )
    
    orientation = Orientation(
        nose=gooey_args.get("nose_orientation"),
        vnc=gooey_args.get("vnc_orientation"),
    )
    
    subject_metadata = SubjectMetadata(
        genotype=gooey_args.get("subject_strain"),
        treatment_details=treatment_details,
        orientation=orientation,
        num_eggs=gooey_args.get("num_eggs", 0),
        notes=gooey_args.get("experimental_notes"),
    )
    
    dev_options = DevOptions(
        prefill_wb_ops=gooey_args.get("prefill_wb_ops", False),
        send_sms_on_completion=gooey_args.get("send_sms", False),
    )
    
    experiment_config = ExperimentConfig(
        experiment_name=gooey_args.get("experiment_name", "default_experiment"),
        output_dir=gooey_args.get("output_folder", "./data"),
        save_images=not gooey_args.get("no_save_images", False),
        save_metadata=not gooey_args.get("no_save_metadata", False),
        save_mip_video=gooey_args.get("save_mip", False),
        acquisition=acquisition_config,
        subject=subject_metadata,
        z_step_size_um=gooey_args.get("z_step_size", 1.0),
        input_recording_path=gooey_args.get("input_recording"),
        dev_options=dev_options,
    )
    
    # Build AlgorithmConfig
    algorithm_params = AlgorithmParameters(
        stim_threshold_pos=gooey_args.get("stim_threshold_pos", 0.06),
        stim_threshold_neg=gooey_args.get("stim_threshold_neg", 0.06),
        stim_cooldown_frames=gooey_args.get("stim_cooldown", 900),
        skip_stimulation_probability=gooey_args.get("skip_stimulation_probability", 0.1),
        delay_stimulation_probability=gooey_args.get("delay_stimulation_probability", 0.4),
        stim_delay_frames_options=gooey_args.get("stim_delay_frames_options", [200, 400]),
        stim_onset_list=gooey_args.get("stim_onset_list_options", []),
        stimulus_diameter_pixels=gooey_args.get("stimulus_diameter", 10),
    )
    
    stimulus_params = StimulusParameters(
        duration_frames_options=gooey_args.get("frames_to_stimulate_for_options", [48]),
        intensity_percent_options=gooey_args.get("stim_intensity_options", [10]),
    )
    
    algorithm_config = AlgorithmConfig(
        algorithm_type=gooey_args.get("trigger_algorithm", "dummy"),
        gui_mode=gooey_args.get("GUI_mode", "neural_imaging"),
        save_algorithm_plot=gooey_args.get("save_alg_model_plot", False),
        algorithm_params=algorithm_params,
        stimulus_params=stimulus_params,
    )
    
    return {
        "hardware": hardware_config,
        "experiment": experiment_config,
        "algorithm": algorithm_config,
    }


def run_acquisition(args: dict[str, Any]):
    """
    Legacy wrapper for backwards compatibility.
    
    Args:
        args: Dictionary containing 'gooey_args' key with configuration
    """
    gooey_args = args.get("gooey_args", {})
    configs = convert_gooey_args_to_configs(gooey_args)
    engine = ClosedLoopEngine(
        hardware_config=configs["hardware"],
        experiment_config=configs["experiment"],
        algorithm_config=configs["algorithm"]
    )
    engine.run()


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
        AlgorithmParameters,
        StimulusParameters,
    )
    
    hardware_config = HardwareConfig(
        backend="dummy",
        stim_interface="dummy",
        microscope_name="test",
    )
    
    acquisition_config = AcquisitionConfig(
        num_frames=100,
        z_planes=10,
        z_step=1.0,
    )
    
    subject_metadata = SubjectMetadata(
        genotype="test_strain",
        notes="Test run with dummy objects",
    )
    
    dev_options = DevOptions(
        prefill_wb_ops=False,
        send_sms_on_completion=False,
    )
    
    experiment_config = ExperimentConfig(
        experiment_name="test_experiment",
        output_dir="./test_output",
        save_images=False,
        save_metadata=False,
        acquisition=acquisition_config,
        subject=subject_metadata,
        dev_options=dev_options,
    )
    
    algorithm_params = AlgorithmParameters(
        stimulus_diameter_pixels=10,
    )
    
    stimulus_params = StimulusParameters(
        enabled=False,
    )
    
    algorithm_config = AlgorithmConfig(
        algorithm_type="dummy",
        enable_gui=False,
        algorithm_params=algorithm_params,
        stimulus_params=stimulus_params,
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