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
from typing import Any

import numpy as np

# Custom libraries and utils
from lib import MMSubroutines
from lib import StimBaseClass
from lib import wbliveUtils as utils


logger = logging.getLogger(__name__)


class ClosedLoopEngine:
    """
    Main engine for closed-loop microscopy acquisition.
    
    Manages the full lifecycle of an acquisition session including:
    - Hardware initialization (microscope, camera)
    - Algorithm setup (trigger detection)
    - Stimulus interface
    - Real-time acquisition loop
    - Data saving and cleanup
    """
    
    def __init__(self, gooey_args: dict[str, Any]):
        """
        Initialize the closed-loop engine.
        
        Args:
            gooey_args: Main configuration from GUI/command line
        """
        self.gooey_args = gooey_args
        
        # Build args structure that matches original format expected by MMSubroutines
        self.args = {"gooey_args": gooey_args}
        
        # Execution state
        self.is_running = False
        self.frame_count = 0
        self.img_count = 0
        self.cooldown_counter = 0
        
        # Components (initialized later)
        self.mmc = None
        self.xsize = 200 # default for unit tests
        self.ysize = 200 # default for unit tests
        self.roi = (0, 0, 200, 200)
        self.args["id"] = 11111111-11-11-11
        self.args["saveroot"] = 'C:/Users/rldun/Downloads/'
        self.alg = None
        self.stim = None
        self.t0 = 1. # default for unit tests
        
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
        
    def _extract_parameters(self):
        """Extract frequently used parameters from configs."""
        self.zsize = self.gooey_args.get("zsize", 1)
        self.frames_to_grab = self.gooey_args.get("total_frames", 100)
        self.frames_baseline_window = self.gooey_args.get("rec_baseline", 0)
        self.no_save_images = self.gooey_args.get("no_save_images", False)
        self.no_save_metadata = self.gooey_args.get("no_save_metadata", False)
        self.save_mip_movie = self.gooey_args.get("save_mip", False)
        self.save_alg_model_plot = self.gooey_args.get("save_alg_model_plot", False)
        self.strobe_acquisition = self.gooey_args.get("strobe_acquisition", False)
        self.strobe_inter_frame_interval = self.gooey_args.get("strobe_inter_frame_interval", 30)
        self.config_file = self.gooey_args.get("mm_configuration_file", "")
        self.is_demo_acquisition = self.config_file.endswith("MMConfig_demo.cfg")
        self.save_structural_scan = self.gooey_args.get("save_structural_scan", "")
        self.prefill_wb_ops = self.gooey_args.get("prefill_wb_ops", False)
        self.notify_sms_on_done = self.gooey_args.get("send_sms", True)
        self.trigger_alg = self.gooey_args.get("trigger_algorithm", "DummyAlg")
        self.acquisition_backend = self.gooey_args.get("acquisition_backend", "micromanager")
        self.gui_mode = self.gooey_args.get("GUI_mode", "neural_imaging")
        
    def initialize_hardware(self):
        """
        Initialize microscope and camera hardware.
        
        Sets up MicroManager core and configures ROI settings.
        """
        logger.info("Initializing hardware...")
        
        # Initialize MicroManager core
        self.mmc = MMSubroutines.initialize_mmc(self.args, self.config_file)
        
        # Configure ROI based on backend
        res = self.mmc.getROI()
        
        # pycromanager returns java objects
        if self.acquisition_backend == "pycromanager":
            self.roi = [res.getX(), res.getY(), res.getWidth(), res.getHeight()]
        else:
            self.roi = res
        self.args["roi"] = self.roi
        
        # grab xsize and ysize form hardware
        self.xsize = self.roi[2]
        self.ysize = self.roi[3]
        
        logger.info(f"Hardware initialized with ROI: {self.roi}")
        
    def initialize_algorithm(self):
        """
        Initialize the closed-loop trigger algorithm.
        
        Uses AlgorithmFactory pattern to instantiate the correct algorithm
        based on configuration.
        """
        logger.info(f"Initializing algorithm: {self.trigger_alg}")
        
        # Algorithm selection
        try:
            if self.trigger_alg == "Dynamic range deriv":
                from lib import DynamicRangeDeriv
                self.alg = DynamicRangeDeriv.DynamicRangeDeriv(self.args)
            elif self.trigger_alg == "RoiDeriv":
                from lib import RoiDeriv
                self.alg = RoiDeriv.RoiDeriv(self.args)
            elif self.trigger_alg == "StimOnsetFromList":
                from lib import StimOnsetFromList
                self.alg = StimOnsetFromList.StimOnsetFromList(self.args)
            elif self.trigger_alg == "PointAndClick":
                from lib import PointAndClick
                self.alg = PointAndClick.PointAndClick(self.args)
            elif self.trigger_alg == "HammerOfDawn":
                from lib import HammerOfDawn
                self.alg = HammerOfDawn.HammerOfDawn(self.args)
            elif self.trigger_alg == "Brainalyzer":
                from lib import Brainalyzer
                self.alg = Brainalyzer.Brainalyzer(self.args, local_handles={"mmc": self.mmc})
            else:
                logger.debug("Running closed-loop with DUMMY algorithm")
                from lib import DummyAlg
                self.alg = DummyAlg.DummyAlg()
                
            # Initialize the algorithm's internal model
            self.alg.initialize_model()
            
        except Exception as err:
            logger.exception(f"Error initializing algorithm: {err}")
            raise
            
        logger.info("Algorithm initialized successfully")
        
    def initialize_stimulus(self):
        """
        Initialize the stimulus interface.
        
        Sets up hardware/software for delivering stimuli based on trigger events.
        """
        logger.info("Initializing stimulus interface...")
        
        try:
            self.stim = StimBaseClass.StimBaseClass.initialize_stim_interface(
                self.args, 
                local_handles={"mmc": self.mmc}
            )
        except Exception as err:
            logger.exception(f"Error initializing stimulus interface: {err}")
            raise
            
        logger.info("Stimulus interface initialized")
        
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
        base_savedir = self.gooey_args.get("output_folder", "./output")
        self.savedir = os.path.join(base_savedir, dt)
        os.makedirs(self.savedir, exist_ok=True)
        
        self.saveroot = os.path.join(self.savedir, dt)
        self.session_id = dt
        
        # Update args dict with session info (needed by MMSubroutines and other components)
        self.args["id"] = self.session_id
        self.args["saveroot"] = self.saveroot
        
        # Initialize frame storage, requires hardware initialization (ysize, xsize)
        self.frames = np.zeros(
            (self.frames_to_grab, self.ysize, self.xsize), 
            dtype=np.uint16
        )
        self.frame_time_list = []
        
        # Configure microscope for acquisition
        MMSubroutines.prepare_live_acquisition(self.mmc, self.args)
        
        # Run pre-acquisition structural scan if requested
        MMSubroutines.run_structural_scan(
            self.save_structural_scan,
            self.mmc,
            self.args,
            self.saveroot,
            self.session_id,
            self.zsize
        )
        
        logger.info(f"Acquisition prepared. Saving to: {self.savedir}")
        
    def run_acquisition_loop(self):
        """
        Execute the main acquisition loop.
        
        Continuously:
        1. Snap images from camera
        2. Process frames through algorithm
        3. Check for stimulus triggers
        4. Submit stimuli if triggered
        5. Save data
        
        Raises:
            Exception: When acquisition is complete or error occurs
        """
        logger.info(f"Starting acquisition loop for {self.frames_to_grab} frames...")
        
        self.is_running = True
        self.img_count = 0
        self.cooldown_counter = 0
        self.t0 = time.time()
        
        false_grab_count = 0
        
        # Start acquisition based on mode
        if self.strobe_acquisition:
            self.next_call = time.time()
            self.mmc.snapImage()
        else:
            frame_grab_t0 = time.time()
            self.mmc.stopSequenceAcquisition()
            self.mmc.clearCircularBuffer()
            self.mmc.startContinuousSequenceAcquisition(0)
            
        try:
            while self.is_running:
                rem = self.mmc.getRemainingImageCount()
                
                while rem > 0 or self.strobe_acquisition:
                    # Grab image from buffer
                    try:
                        if self.strobe_acquisition:
                            img = self.mmc.getImage().astype(np.uint16)
                        else:
                            img = self.mmc.popNextImage().astype(np.uint16)
                            self.next_call = frame_grab_t0
                    except Exception as err:
                        false_grab_count += 1
                        logger.debug(f"False grab #{false_grab_count}: {err}")
                        continue
                        
                    # Reshape if using pycromanager
                    if self.acquisition_backend == "pycromanager":
                        img = img.reshape((self.ysize, self.xsize))
                        
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
                        
                    # Handle cooldown
                    if self.cooldown_counter > 0:
                        self.cooldown_counter -= 1
                        
                    # Process frame through algorithm
                    zndx = image_ndx % self.zsize
                    self.alg.process_frame(img, zndx)
                    
                    # Check for stimulus trigger
                    stim_params, self.cooldown_counter = self.alg.check_stim(
                        image_ndx, self.cooldown_counter
                    )
                    
                    # Submit stimulus if triggered
                    self.stim.submit_stim_params(stim_params, image_ndx)
                    
                    # Volume completion handling
                    if zndx == self.zsize - 1:
                        self.stim.check_stim(self.img_count)
                        
                    # Check if acquisition complete
                    if self.img_count == self.frames_to_grab:
                        raise Exception("Acquisition complete!")
                        
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
                            
                        self.mmc.snapImage()
                    else:
                        rem = self.mmc.getRemainingImageCount()
                        
        except Exception as err:
            logger.info(f"Acquisition loop ended: {err}")
        finally:
            self.is_running = False
            t1 = time.time()
            logger.info(f"Acquisition complete. Duration: {t1 - self.t0:.2f}s")
            
    def save_metadata(self):
        """
        Collect and save metadata from all components.
        
        Aggregates metadata from:
        - Algorithm
        - Stimulus interface
        - MicroManager
        - Acquisition settings
        """
        if self.no_save_metadata:
            logger.info("Metadata saving disabled")
            return
            
        logger.info("Saving metadata...")
        
        # Build metadata dict - using self.args as base which has the expected structure
        metadata = dict(self.args)
        metadata["frame_time_list"] = self.frame_time_list
        metadata["t0"] = self.t0
        metadata["xsize"] = self.xsize
        metadata["ysize"] = self.ysize
        
        # Collect component metadata
        if self.alg:
            metadata["alg_metadata"] = self.alg.get_metadata(args=metadata)
            
        if self.stim:
            metadata["stim_metadata"] = self.stim.get_metadata(args=metadata)
            
        if self.mmc:
            metadata["mmc_metadata"] = MMSubroutines.get_metadata(
                args=metadata, mmc=self.mmc
            )
            
        # Save to file
        utils.save_metadata(
            savefilename=self.saveroot + "_metadata.json",
            metadata=metadata
        )
        
        # Prefill wb_ops if requested
        if self.prefill_wb_ops:
            utils.prefill_wb_ops(savefileroot=self.savedir, metadata=metadata)
            
        logger.info("Metadata saved")
        
    def _save_images(self):
        """Save acquired image stack."""
        if self.no_save_images:
            logger.info("Image saving disabled")
            return
            
        logger.info("Saving images...")
        MMSubroutines.saveScanTiffs(
            fname=self.saveroot + ".tiff",
            img_array=self.frames
        )
        logger.info("Images saved")
        
    def _save_visualizations(self):
        """Save algorithm plots and MIP movies."""
        if self.save_alg_model_plot and self.alg:
            logger.info("Saving algorithm model plot...")
            self.alg.plot_model(savefilename=self.saveroot + "_live_stim_fig.svg")
            
        if self.save_mip_movie:
            logger.info("Generating MIP movie...")
            exposure = self.gooey_args.get("exposure", 30)
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
        
        Stops acquisition, closes hardware interfaces, and sends notifications.
        """
        logger.info("Cleaning up...")
        
        # Stop hardware
        if self.mmc:
            try:
                self.mmc.stopSequenceAcquisition()
            except Exception as err:
                logger.warning(f"Error stopping sequence acquisition: {err}")
                
        # Close algorithm
        if self.alg:
            try:
                self.alg.close()
            except Exception as err:
                logger.warning(f"Error closing algorithm: {err}")
                
        # Close stimulus interface
        if self.stim:
            try:
                self.stim.close()
            except Exception as err:
                logger.warning(f"Error closing stimulus interface: {err}")
                
        # Close MicroManager
        if self.mmc:
            try:
                MMSubroutines.close(self.mmc, self.args)
            except Exception as err:
                logger.warning(f"Error closing MicroManager: {err}")
                
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
        1. Prepare acquisition (create dirs, etc.)
        2. Initialize hardware
        3. Initialize algorithm
        4. Initialize stimulus
        5. Run acquisition loop
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
            self.prepare_acquisition() # reads from hardware, initializes values for alg/stim
            self.initialize_algorithm()
            self.initialize_stimulus()
            
            
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
        # Handle simulation mode
        # TODO: Implement simulation support
        return
        
    # Run acquisition
    try:
        engine = ClosedLoopEngine(gooey_args=ops)
        engine.run()
    except Exception as err:
        logger.exception(f"Error running acquisition: {err}")
        raise
    finally:
        logger.info("Session complete")
        sys.exit()


def run_acquisition(args: dict[str, Any]):
    """
    Legacy wrapper for backwards compatibility.
    
    Args:
        args: Dictionary containing 'gooey_args' key with configuration
    """
    gooey_args = args.get("gooey_args", {})
    engine = ClosedLoopEngine(gooey_args=gooey_args)
    engine.run()



def create_test_config(input_recording: str | None = None) -> dict[str, Any]:
    """
    Create a test configuration for running the engine with dummy objects.
    
    Args:
        input_recording: Optional path to a TIFF file for simulated acquisition.
                        If None, will use DummyMMC without data.
    
    Returns:
        Dictionary with test configuration matching gooey_args format
    """
    test_config = {
        # Acquisition controls
        "output_folder": "./test_output",
        "total_frames": 100,  # Small number for quick testing
        "mm_configuration_file": "MMConfig_demo.cfg", # Not used
        "zsize": 10,
        "save_mip": False,
        "strobe_acquisition": False,
        "strobe_inter_frame_interval": 80,
        "save_structural_scan": "none",
        
        # Experimental metadata (minimal for testing)
        "subject_strain": "test_strain",
        "subject_condition": "",
        "atr_concentration": 0.0,
        "z_step_size": 3.0,
        "nose_orientation": "left",
        "vnc_orientation": "up",
        "num_eggs": 0,
        "microscope_name": "test",
        "experimental_notes": "Test run with dummy objects",
        
        # Closed-loop controls
        "trigger_algorithm": "Dummy algorithm (does nothing)",
        "GUI_mode": "neural_imaging",
        "rec_baseline": 0,
        "save_alg_model_plot": False,
        
        # Stimulus settings
        "stim_interface": "no stim",
        "use_static_stim_roi": False,
        "frames_to_stimulate_for_options": [48],
        "stim_intensity_options": [10],
        "stimulus_diameter": 10,
        
        # Dev ops
        "input_recording": input_recording,
        "acquisition_backend": "test",
        "no_save_images": True,  # Don't save images during testing
        "no_save_metadata": True,  # Don't save metadata during testing
        "save_gooey_defaults": False,
        "prefill_wb_ops": False,
        "send_sms": False,
        
        # Additional params that might be needed
        "roi": [0, 0, 512, 512],
        "exposure": 30,
        "binning": "1x1",
        "configs": {},
    }
    
    return test_config


def run_test():
    """
    Run a test acquisition using dummy objects.
    
    This function demonstrates how to run the ClosedLoopEngine with
    stub objects for testing without real hardware.
    """
    print("="*60)
    print("Running ClosedLoopEngine Test")
    print("="*60)
    
    # Option 1: Test with a real TIFF file for realistic simulation
    # Uncomment and provide path to test with actual data:
    # test_config = create_test_config(input_recording="path/to/your/test.tiff")
    
    # Option 2: Test with dummy objects (no real data)
    test_config = create_test_config()
    
    # Create and run engine
    try:
        engine = ClosedLoopEngine(gooey_args=test_config)
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

    # in parent directory, run:
    # python -m engine.closed_loop_engine

    # Run the test when this module is executed directly
    import sys
    
    # Set up logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if a TIFF file was provided as command line argument
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Running test with input file: {test_file}")
        test_config = create_test_config(input_recording=test_file)
    else:
        print("Running test with dummy objects (no input file)")
        test_config = create_test_config()
    
    # Run test
    success = run_test()
    sys.exit(0 if success else 1)