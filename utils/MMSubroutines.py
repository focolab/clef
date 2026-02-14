import time
import sys
import numpy as np
import tifffile as tf
from datetime import datetime
import logging
import os
# import serial

import time
import json
from config.config_manager import HardwareConfig

# try:
#     from lib import DummyMMC
# except ImportError as err:
#     # change path and try again
#     import DummyMMC
from hardware.backends.lib import DummyMMC

logger = logging.getLogger(__name__)


def initialize_mmc(
    # args,
    # config_file="C:\\Program Files\\Micro-Manager-2.0beta\\190814_PRIME_BSI_TIE.cfg",
    # acquisition_backend='dummy',
    # config_file="C:\\Program Files\\Micro-Manager-2.0beta\\200212_PRIME_BSI_TIE.cfg",
    # dummy_fname=None,
    config: HardwareConfig

):

    # acquisition_backend = args["gooey_args"]["acquisition_backend"]
    # microscope_name = args["gooey_args"]["microscope_name"]

    acquisition_backend = config.backend_configuration


    if acquisition_backend == 'test' or acquisition_backend == 'dummy':
        return DummyMMC.DummyMMC()

    elif acquisition_backend == "pycromanager":

        # simple single image acquisition example with snap

        #### Setup ####
        # if microscope_name == "innovation core spinning disk":
        # from pycromanager import Bridge
        # bridge = Bridge(convert_camel_case=False)
        # mmc = bridge.get_core()
        # elif microscope_name == "innovation core thunderscope":
        from pycromanager import Core, JavaObject
        mmc = Core(convert_camel_case=False)

        return mmc
    
    else:
        logger.critical(f'Attempting to initialize mmc object with unrecognized backend {acquisition_backend}')

    # else:

    #     # if we're doing a fake run through of data, return a fake MMC object
    #     # if args["gooey_args"]["input_recording"] is not None:
    #     if hasattr(config.backend_configuration.input_recording):
    #         # dummy_fname = args["gooey_args"]["input_recording"]
    #         logging.debug(f"Creating dummy MMC object from {config.backend_configuration.input_recording}")
    #         return DummyMMC.DummyMMC(config.backend_configuration.input_recording)

    #     # initialize core object and camera, all hardcoded
    #     scope = args["gooey_args"]["microscope_name"]
    #     if scope == "torstoscope spinning disk":
    #         path_to_mm = "C:\\Program Files\\Micro-Manager-2.0beta"
    #         sys.path.append(path_to_mm)
    #         try:
    #             import MMCorePy

    #             mmc = MMCorePy.CMMCore()

    #         except ImportError as err:
    #             logging.exception(
    #                 "Unable to import Micro-Manager python bindings: {}".format(err)
    #             )
    #             raise (err)

    #         # load configuration
    #         try:
    #             logging.debug("Loading configuration file: {}".format(config_file))
    #             mmc.loadSystemConfiguration(config_file)

    #         except Exception as err:
    #             logging.warning(
    #                 "Error during configuration loading: {} \nTrying again...".format(
    #                     err
    #                 )
    #             )
    #             time.sleep(10)
    #             mmc = MMCorePy.CMMCore()
    #             mmc.loadSystemConfiguration(config_file)

    #     elif scope == "innovation core spinning disk":
    #         import pymmcore

    #         # set where to find device adapters and change to mm parent dir (to prevent silent crashes)
    #         mm_dir = "C:\\Program Files\\Micro-Manager-2.0"
    #         curr_path = os.getcwd()
    #         os.chdir(mm_dir)
    #         mmc = pymmcore.CMMCore()
    #         mmc.setDeviceAdapterSearchPaths([mm_dir])

    #         # load configuration
    #         try:

    #             logging.debug("Loading configuration file: {}".format(config_file))
    #             mmc.loadSystemConfiguration(config_file)
    #             os.chdir(curr_path)

    #         except Exception as err:
    #             logging.warning(
    #                 "Error during configuration loading: {} \nTrying again...".format(
    #                     err
    #                 )
    #             )
    #             time.sleep(10)
    #             os.chdir(mm_dir)
    #             mmc = MMCorePy.CMMCore()
    #             mmc.loadSystemConfiguration(config_file)
    #             os.chdir(curr_path)
    #     return mmc


# def prepare_live_acquisition(mmc, args):
#     # configure microsope state
#     # NOTE: Device-specific properties (setProperty, setConfig, setAutoShutter, etc.)
#     # are now handled by MicroManagerBackend during initialization via HardwareConfig.
#     # This function now only handles Z-stack configuration and other acquisition-specific setup.
#     try:

#         # first check acquisition backend
#         acquisition_backend = args["gooey_args"]["acquisition_backend"]
#         scope = args["gooey_args"]["microscope_name"]

#         if acquisition_backend == "pycromanager":

#             if scope == "innovation core spinning disk":

#                 # Z-stack stage configuration should be done via StageInterface.configure_stage()
#                 # or StageInterface.run_z_stack() before calling prepare_live_acquisition().
#                 # Example:
#                 #   stage.configure_stage({
#                 #       "z_start": -(numz - 1) * zStepSize / 2,
#                 #       "z_end": (numz - 1) * zStepSize / 2,
#                 #       "z_step": zStepSize,
#                 #       "pad_z": 0
#                 #   })
#                 # 
#                 # For backward compatibility, if zsize > 1, log a warning that stage
#                 # should be configured via stage interface.
#                 if args["gooey_args"]["zsize"] > 1:
#                     logging.warning(
#                         "Z-stack detected (zsize > 1). Stage configuration should be done "
#                         "via StageInterface.configure_stage() or StageInterface.run_z_stack() "
#                         "before calling prepare_live_acquisition(). Using deprecated "
#                         "set_asi_stage_buffer() for backward compatibility."
#                     )
#                     numz = args["gooey_args"]["zsize"]
#                     zStepSize = float(args["gooey_args"]["z_step_size"])
#                     set_asi_stage_buffer(
#                         mmc,
#                         zStart=-(numz - 1) * zStepSize / 2,
#                         zEnd=(numz - 1) * zStepSize / 2,
#                         zStepSize=zStepSize,
#                         padZ=0,
#                     )

#                 # Device-specific properties (auto_shutter, device properties, shutters)
#                 # are now configured via HardwareConfig in MicroManagerBackend.initialize()
#                 # No need to set them here.

#             elif scope == "innovation core thunderscope":
#                 # Device-specific properties are now configured via HardwareConfig
#                 # Only Z-stack or other acquisition-specific logic should remain here
#                 pass

#         else:
#             # pymmcore backend
#             if scope == "torstoscope spinning disk":

#                 # Device-specific properties (exposure, binning, auto_shutter, shutters, configs)
#                 # are now configured via HardwareConfig in MicroManagerBackend.initialize()
#                 # Only acquisition-specific logic remains here.

#                 # Set ROI from args (this is acquisition-specific, not device initialization)
#                 roi = args["roi"]
#                 mmc.setROI(roi[0], roi[1], roi[2], roi[3])

#                 # Note: Device-specific properties and configs are now configured via
#                 # HardwareConfig in MicroManagerBackend.initialize(). See:
#                 # - config/hardware_config_torstoscope.yaml for torstoscope settings
#                 # - config/hardware_config_innovation_core.yaml for innovation core settings
#                 #
#                 # Config presets from args["configs"] are deprecated and kept only for
#                 # backward compatibility. New code should use HardwareConfig.devices[].configs
#                 configs = args.get("configs", {})
#                 if configs:
#                     logging.warning(
#                         "Using deprecated args['configs'] for device configuration. "
#                         "Migrate to HardwareConfig.devices[].configs in YAML config file."
#                     )
#                     for k in configs:
#                         if k == ">>> Presets":
#                             continue
#                         try:
#                             if type(configs[k]) == dict:
#                                 # Corner case: DAC voltage settings
#                                 label = "DAC{}".format(k[:-2])
#                                 prop = "Volts"
#                                 val = configs[k][prop]
#                                 mmc.setProperty(label, prop, float(val))
#                             else:
#                                 mmc.setConfig(k, configs[k])
#                         except Exception as e:
#                             logging.warning(f"Could not set config {k}: {e}")

#                 # Set stage moving with serial command (acquisition-specific)
#                 if args["gooey_args"]["zsize"] > 1:
#                     port = "COM11"
#                     command = "2"
#                     endln = "\r"
#                     try:
#                         mmc.setSerialPortCommand(port, command, endln)
#                     except Exception as e:
#                         logging.warning(f"Could not set serial port command: {e}")

#             elif scope == "innovation core spinning disk":

#                 raise (Exception("ERROR: pymmcore structural images not tested"))

#                 # Device-specific properties are now configured via HardwareConfig
#                 # Only acquisition-specific logic should remain here.

#                 # Set ROI from args
#                 roi = args["roi"]
#                 mmc.setROI(roi[0], roi[1], roi[2], roi[3])

#                 # Note: Device-specific properties and configs are now configured via
#                 # HardwareConfig in MicroManagerBackend.initialize(). See:
#                 # - config/hardware_config_torstoscope.yaml for torstoscope settings
#                 # - config/hardware_config_innovation_core.yaml for innovation core settings
#                 #
#                 # Config presets from args["configs"] are deprecated and kept only for
#                 # backward compatibility. New code should use HardwareConfig.devices[].configs
#                 configs = args.get("configs", {})
#                 if configs:
#                     logging.warning(
#                         "Using deprecated args['configs'] for device configuration. "
#                         "Migrate to HardwareConfig.devices[].configs in YAML config file."
#                     )
#                     for k in configs:
#                         try:
#                             if type(configs[k]) == dict:
#                                 # Corner case: DAC voltage settings
#                                 label = "DAC{}".format(k[:-2])
#                                 prop = "Volts"
#                                 val = configs[k][prop]
#                                 mmc.setProperty(label, prop, float(val))
#                             else:
#                                 mmc.setConfig(k, configs[k])
#                         except Exception as e:
#                             logging.warning(f"Could not set config {k}: {e}")

#                 # Z-stack stage configuration should be done via StageInterface.configure_stage()
#                 # or StageInterface.run_z_stack() before calling prepare_live_acquisition().
#                 if args["gooey_args"]["zsize"] > 1:
#                     logging.warning(
#                         "Z-stack detected (zsize > 1). Stage configuration should be done "
#                         "via StageInterface.configure_stage() or StageInterface.run_z_stack() "
#                         "before calling prepare_live_acquisition(). Using deprecated "
#                         "set_asi_stage_buffer() for backward compatibility."
#                     )
#                     numz = args["gooey_args"]["zsize"]
#                     zStepSize = float(args["gooey_args"]["z_step_size"])
#                     set_asi_stage_buffer(
#                         mmc,
#                         zStart=-(numz - 1) * zStepSize / 2,
#                         zEnd=(numz - 1) * zStepSize / 2,
#                         zStepSize=3,
#                         padZ=0,
#                     )

#     except Exception as err:
#         logging.exception(
#             "Error during MMC configuration setting for scope {} with backend {}: {}".format(
#                 scope, acquisition_backend, err
#             )
#         )
#         raise (err)


# def close(mmc, args):

#     # stop possible running acquisition
#     try:
#         mmc.stopSequenceAcquisition()

#         # if relevant stop stage or triggerscope sequence
#         scope = args["gooey_args"]["microscope_name"]
#         if scope == "innovation core spinning disk":

#             logging.info("Halting stage and laserTTL property sequences")

#             # hardcoded
#             laserTTLs = "TTL1-8"
#             stage = mmc.getFocusDevice()
#             mmc.stopStageSequence(stage)
#             mmc.waitForDevice(stage)
#             mmc.setPosition(stage, 0)
#             mmc.waitForDevice(stage)
#             mmc.stopPropertySequence(laserTTLs, "State")

#             # note that these get re-set if we run another structural scan upon imaging completion

#     # fail clunkily :)
#     except Exception as err:
#         logging.warning("Error during MMSubroutines.close(): {}".format(err))


# function to pull various settings and add them into metadata
# def get_metadata(args, mmc):

#     scope = args["gooey_args"]["microscope_name"]
#     # acquisition_backend = args["gooey_args"]["acquisition_backend"]

#     # if acquisition backend is pycromanager we need to grab settings because they weren't already sent to wb-live
#     cam = mmc.getCameraDevice()
#     binning = mmc.getProperty(cam, "Binning")

#     # metadata
#     metadata = {
#         "binning": binning,
#         "exposure": mmc.getExposure(),
#     }

#     if scope == 'innovation core spinning disk':
#         intensity_405 = mmc.getProperty("DAC405", "Volts")
#         intensity_488 = mmc.getProperty("DAC488", "Volts")
#         intensity_561 = mmc.getProperty("DAC561", "Volts")
#         intensity_639 = mmc.getProperty("DAC639", "Volts")
#         more_metadata = {
#             "camera_mode": mmc.getCurrentConfig("Camera Mode"),
#             "objective": mmc.getCurrentConfig("Objective"),
#             "intensity_488": intensity_488,
#             "intensity_561": intensity_561,
#             "intensity_405": intensity_405,
#             "intensity_639": intensity_639,
#         }
#         metadata.update(more_metadata)


#     return metadata


# def prepare_demo_acquisition(mmc, args):

#     # if dealing with dummy mmc object, don't do anything
#     try:
#         if mmc.is_dummy:
#             return
#     except Exception:
#         pass

#     # set params from args
#     # first check acquisition backend
#     acquisition_backend = args["gooey_args"]["acquisition_backend"]
#     if acquisition_backend == "pycromanager":
#         return
#     else:
#         cam = "Camera"
#         mmc.setExposure(args["exposure"])
#         mmc.setProperty(cam, "Binning", args["binning"])
#         mmc.setCircularBufferMemoryFootprint(10000)
#         mmc.setAutoShutter(True)
#         roi = args["roi"]
#         mmc.setROI(roi[0], roi[1], roi[2], roi[3])


# function to do brief recording, was necessary a while ago because of a weird bug
# where laser wasn't actually triggered. i think the problem was with autoshutter and
# i haven't troubleshot in a while
# def pre_flight_check(mmc, zsize):
#     """quick run through to test flight"""

#     try:

#         # start and go because for some reason that's a weird bug with the autoshutter
#         logging.debug("Performing start and stop...")
#         # mmc.startContinuousSequenceAcquisition(0)
#         mmc.startSequenceAcquisition(zsize, 0, False)
#         while mmc.isSequenceRunning():
#             pass

#         # wait a sec and clear data acquired
#         # i don't remember if the hard pause here is necessary
#         time.sleep(0.2)
#         mmc.clearCircularBuffer()
#         time.sleep(0.2)

#     except Exception as err:
#         logging.exception("Error during pre-flight-check! {}".format(err))
#         raise (err)


# def run_structural_scan(save_structural_scan, mmc, args, saveroot, dt, zsize):

#     # grab zsize
#     zsize = int(args["gooey_args"]["zsize"])
#     z_step_size = int(args["gooey_args"]["z_step_size"])

#     # no structural scan
#     if save_structural_scan == "none":
#         return

#     # set dir to save rec
#     structural_scan_dir = saveroot + "_structuralScan_0"

#     # if path to structural scan already exists, we're probably taking a rec at the end of the rec
#     if not os.path.isdir(structural_scan_dir):

#         structural_scan_root = structural_scan_dir + "\\{}".format(dt)
#         logging.debug(
#             "Structural scan output directory: {}".format(structural_scan_dir)
#         )
#         os.mkdir(structural_scan_dir)
#         args["structural_scan_dir"] = structural_scan_dir

#     else:

#         # set a second structural scan directory, increment suffix
#         structural_scan_dir = structural_scan_dir[:-1] + "1"
#         structural_scan_root = structural_scan_dir + "\\{}".format(dt)
#         logging.debug(
#             "Structural scan output directory: {}".format(structural_scan_dir)
#         )

#         # make new directory
#         os.mkdir(structural_scan_dir)
#         first_structural_scan = args["structural_scan_dir"]

#         # store reference to second structural scan
#         args["structural_scan_dir"] = [first_structural_scan, structural_scan_dir]

#     # gfp + rfp standard filters
#     if "GFP + RFP (torstoscope)" in save_structural_scan:

#         # if we want a higher resolution structural scan we can send an updated serial port program
#         if "1um steps" in save_structural_scan:

#             # upload finer resolution stage program to asi stage
#             stage_program_name = "z = {} x 1um".format(int(zsize * z_step_size))
#             upload_asi_stage_program(stage_program_name, mmc, args)

#             # edit zsize for structural scans
#             zsize = int(zsize * z_step_size)

#         # subroutine to do structural scans
#         ch_list = ["488", "561", "Brightfield"]
#         for ch in ch_list:
#             structural_scan_channel(
#                 mmc=mmc,
#                 lightsource=ch,
#                 fname_root=structural_scan_root,
#                 zsize=zsize,
#                 args=args,
#             )

#         # reset higher resolution structural scan params
#         if "1um steps" in save_structural_scan:

#             # zsize was edited in the above if statement, so reset it here
#             zsize = int(zsize / z_step_size)

#             # upload 12x3um program, assuming that's what we want
#             stage_program_name = "z = {} x 3um".format(zsize)
#             upload_asi_stage_program(stage_program_name, mmc, args)

#         # reset conditions for imaging after structural scan
#         prepare_live_acquisition(mmc, args)
#         # MMSubroutines.pre_flight_check(mmc, zsize)

#     if "NeuroPAL (innovation core)" in save_structural_scan:

#         # note that first draft of this is very hardcoded
#         # grab current settings to recapitulate after possible changes during structural scan
#         zsize = args["gooey_args"]["zsize"]
#         zStepSize = float(args["gooey_args"]["z_step_size"])

#         # there's probably a way to automate this, but configs in micro-manager
#         # are sometimes blank (after some underlying properties get changed by other
#         # code) so we have to hardcode for now
#         curr_setup = {
#             "props": {
#                 "DAC488": mmc.getProperty("DAC488", "Volts"),
#             },
#             # "configs": {"Mightex-Setup": "640-SP"},
#             "configs": {},
#             "exposure": mmc.getExposure(),
#             "zsize": zsize,
#             "zstepsize": zStepSize,
#             "zposition": mmc.getPosition(),
#         }

#         ## set up hardcoded channel settings
#         # zStepSize = 0.75
#         # zsize = 48
#         zStepSize = 1
#         zsize = 40

#         # upload program using micro-manager api. assumes imaging is in middle
#         set_asi_stage_buffer(
#             mmc,
#             zStart=-(zsize - 1) * zStepSize / 2,
#             zEnd=(zsize - 1) * zStepSize / 2,
#             zStepSize=zStepSize,
#             padZ=0,
#         )

#         # set laser settings
#         mmc.setProperty("DAC405", "Volts", 3.5)
#         mmc.setProperty("DAC488", "Volts", 1.5)
#         mmc.setProperty("DAC561", "Volts", 1.5)
#         mmc.setProperty("DAC639", "Volts", 3.5)
#         # mmc.setProperty("DAC405", "Volts", 2.0)
#         # mmc.setProperty("DAC488", "Volts", 0.5)
#         # mmc.setProperty("DAC561", "Volts", 0.4)
#         # mmc.setProperty("DAC639", "Volts", 2.5)
#         # mmc.setProperty("Transmitted Light", "Level", 30)

#         # set turret filter away from dual opto/imaging
#         # at the moment, motorized condensor hits the chip tubing if we set
#         # IL filter wheel to empty, so instead we can just set to 640 mirror
#         mmc.setConfig("Mightex-Setup", "640-Mirror")
#         # mmc.setConfig("TL Control", "Computer")

#         # turn off ttl property sequence so we don't just trigger 488
#         laserTTLs = "TTL1-8"
#         mmc.stopPropertySequence(laserTTLs, "State")

#         # channels
#         ch_list = [
#             {
#                 "Channel": "405",
#                 "exposure": 150,
#             },
#             {
#                 "Channel": "561-700-75m",
#                 "exposure": 150,
#             },
#             {
#                 "Channel": "488-700-75m",
#                 "exposure": 150,
#             },
#             {
#                 "Channel": "561",
#                 "exposure": 100,
#             },
#             {
#                 "Channel": "488-605-70m",
#                 "exposure": 50,
#             },
#             {
#                 "Channel": "488",
#                 "exposure": 20,
#             },
#             {
#                 "Channel": "639",
#                 "exposure": 150,
#             },

#             #{
#             #    "Channel": "BF",
#             #    "exposure": 100,
#             #}
#         ]

#         # iterate channel lists
#         for ch in ch_list:
#             mmc.setConfig("Channel", ch["Channel"])
#             mmc.waitForConfig("Channel", ch["Channel"])
#             mmc.setExposure(ch["exposure"])

#             # take zscan and save
#             imgs = snapScan(mmc, zsize, args)
#             saveScanTiffs(structural_scan_root + "_{}.tiff".format(ch["Channel"]), imgs)

#         ## reset settings
#         # reset exposure
#         mmc.setExposure(curr_setup["exposure"])

#         # reset stage
#         mmc.setPosition(curr_setup["zposition"])
#         zStepSize = curr_setup["zstepsize"]
#         zsize = curr_setup["zsize"]

#         # upload program using micro-manager api. assumes imaging is in middle
#         set_asi_stage_buffer(
#             mmc,
#             zStart=-(zsize - 1) * zStepSize / 2,
#             zEnd=(zsize - 1) * zStepSize / 2,
#             zStepSize=zStepSize,
#             padZ=0,
#         )

#         # reset properties
#         mmc.setProperty("DAC488", "Volts", curr_setup["props"]["DAC488"])
#         mmc.waitForDevice("DAC488")

#         # reset configs
#         # mmc.setConfig("Mightex-Setup", "640-SP")
#         # mmc.waitForConfig("Mightex-Setup", "640-SP")
#         # mmc.setConfig("Channel", "488")
#         # mmc.waitForConfig("Channel", "488")

#         # note that shutters, triggerscope etc should be set in a future call to
#         # prepare
#         generate_tiffreader(structural_scan_dir, args, mmc)

#     # if "NeuroPAL (innovation core, Muneki settings)":

        


# def generate_tiffreader(structural_scan_dir, args, mmc, zsize=48):

#     # get recording id
#     fname_root = args["id"]

#     # get voxel dimensions
#     zStepSize = float(args["gooey_args"]["z_step_size"])
#     xy = mmc.getPixelSizeUm()

#     # get channels, numz, roi
#     roi = mmc.getROI()
#     height = roi.getHeight()
#     width = roi.getWidth()

#     # build tiffreader json file
#     tr = {
#         "files": [
#             fname_root + "_405.tiff",
#             fname_root + "_488-605-70m.tiff",
#             fname_root + "_488.tiff",
#             fname_root + "_561-700-75m.tiff",
#             fname_root + "_561.tiff",
#             fname_root + "_639.tiff",
#         ],
#         "dtype": "uint16",
#         "axes": "TCZYX",
#         "shape": [1, 6, zsize, height, width],
#         "offset": 0,
#         "pixel_size": {"X": xy, "Y": xy, "Z": zStepSize},
#     }

#     # write tiffreader json file
#     tr_fname = structural_scan_dir + "/tiffreader.json"

#     logging.info('Generating tiffreader along with structural images: {}'.format(tr_fname))
#     with open(tr_fname, "w") as outfile:
#         json.dump(tr, outfile, indent=4, sort_keys=True)


# def upload_asi_stage_program(my_program, mmc, args, stage_port="COM6", stage_baud=9600):
#     """function to upload a asi stage console program from a text file"""

#     # set asi stage com port based on microscope
#     scope = args["gooey_args"]["microscope_name"]
#     if scope == "torstoscope spinning disk":
#         stage_port = "COM6"
#     elif scope == "innovation core spinning disk":
#         stage_port = "todo"

#     # grab zsize
#     zsize = int(args["gooey_args"]["zsize"])

#     # filter input
#     if my_program == "z = 12 x 3um":
#         program_fname = "C:/DATA/RLD/piezo_z_12x3um.txt"
#     elif my_program == "z = 36 x 1um":
#         program_fname = "C:/DATA/RLD/piezo_z_36x1um.txt"
#     elif my_program == "z = 48 x 0.75um":
#         lol = "todo"
#     else:
#         the_problem = "ASI stage program {} not recognized. This could lead to undefined behavior. Does the asi console save file exist?".format(
#             my_program
#         )
#         raise (Exception(the_problem))

#     try:

#         # load stage programmed text file
#         with open(program_fname) as f:
#             program_lines = f.readlines()

#         # load serial connection
#         import serial
#         ser = serial.Serial(stage_port, stage_baud, timeout=0)

#         # loop on individual lines in doc
#         endline = "\r"
#         for line in program_lines:

#             # send with endline, removing \n endline found in text file
#             cmd = line[:-1] + endline
#             ser.write(cmd.encode())

#         # TTL pulse to begin stage progression
#         port = "COM11"
#         command = "2"
#         endln = "\r"
#         mmc.setSerialPortCommand(port, command, endln)

#         # close serial port
#         ser.close()

#     except Exception as err:
#         logging.critical("ERROR while uploading asi stage program: {}".format(err))


# def set_asi_stage_buffer(mmc, zStart=-16.5, zEnd=16.5, zStepSize=3, padZ=0):
#     """
#     Upload ASI stage positions into internal ring buffer through micro-manager api.
    
#     DEPRECATED: This function is kept for backward compatibility only.
#     New code should use StageInterface.configure_stage() or StageInterface.run_z_stack()
#     instead. This provides better abstraction and works with the hardware manager.
    
#     Args:
#         mmc: Micro-Manager Core object
#         zStart: Starting Z position
#         zEnd: Ending Z position
#         zStepSize: Step size
#         padZ: Number of padding steps at start
#     """

#     # hardcoded params
#     laserTTLs = "TTL1-8"
#     gcampTTL = "18"

#     # get hardware
#     stage = mmc.getFocusDevice()
#     # camera = mmc.getCameraDevice()

#     # start by setting position to zero
#     mmc.setPosition(stage, 0)

#     # load bridge for custom java objects
#     # bridge = Bridge(convert_camel_case=False)
#     dv = JavaObject("mmcorej.DoubleVector")
#     dv_list = []
#     sv = JavaObject("mmcorej.StrVector")
#     z = zStart
#     nrSteps = 0

#     # quick semantic check for case of 1Z plane imaging + structural scan
#     # spec loop will hang unless zStepSize is set to some value > 0
#     if zStepSize == 0:
#         zStepSize = 1

#     # pad zstep array with extra steps at zStart
#     for i in range(0, padZ):
#         dv.add(z)
#         dv_list.append(z)
#         sv.add("0")
#         nrSteps += 1

#     # ascending z-steps
#     while z <= zEnd:
#         dv.add(z)
#         dv_list.append(z)
#         sv.add(gcampTTL)
#         z += zStepSize
#         nrSteps += 1

#     logging.info(
#         "Uploading {} zPositions to ASI stage: {}".format(len(dv_list), dv_list)
#     )

#     # upload and configure
#     mmc.setPosition(stage, zStart)
#     mmc.waitForDevice(stage)
#     mmc.stopStageSequence(stage)
#     mmc.loadStageSequence(stage, dv)
#     mmc.stopPropertySequence(laserTTLs, "State")
#     mmc.loadPropertySequence(laserTTLs, "State", sv)
#     mmc.startStageSequence(stage)
#     mmc.startPropertySequence(laserTTLs, "State")


# def structural_scan_channel(mmc, lightsource, fname_root, zsize, args, num_images=None):
#     """Function to take a fixed number of images to establish structural infromation. Assumes that only thing that needs to be changed is filters."""

#     try:

#         scope = args["gooey_args"]["microscope_name"]
#         backend = args["gooey_args"]["acquisition_backend"]

#         if backend == "pycromanager":
#             raise (
#                 Exception(
#                     "pycromanager backend driving structural scans not implemented!"
#                 )
#             )

#         # handle each light source, for various hardwares
#         if lightsource == "488":
#             if scope == "torstoscope spinning disk":
#                 mmc.setConfig("LMM5", "488")
#                 mmc.setConfig("LMM5-488-intensity", "10")
#                 mmc.setConfig("Confocal filter wheel", "Y_488_525")
#             elif scope == "innovation core spinning disk":
#                 mmc.setConfig("Channel", "488")
#                 mmc.setProperty("DAC488", "Volts", 0.5)
#                 mmc.setExposure(10)

#         elif lightsource == "561":
#             if scope == "torstoscope spinning disk":
#                 mmc.setConfig("LMM5", "561")
#                 mmc.setConfig("LMM5-561-intensity", "20")
#                 mmc.setConfig("Confocal filter wheel", "Y_561LP")
#             elif scope == "innovation core spinning disk":
#                 mmc.setConfig("Channel", "561")
#                 mmc.setProperty("DAC561", "Volts", 0.4)
#                 mmc.setExposure(50)

#         elif lightsource == "Brightfield":
#             if scope == "torstoscope spinning disk":
#                 mmc.setConfig("LMM5", "OFF")
#                 mmc.setConfig("DIA", "ON")
#                 mmc.setConfig("Confocal filter wheel", "Empty")
#             elif scope == "innovation core spinning disk":
#                 raise (
#                     Exception(
#                         "structural scan for channel {} on scope {} not implemented!".format(
#                             lightsource, scope
#                         )
#                     )
#                 )

#         elif lightsource == "405":
#             if scope == "torstoscope spinning disk":
#                 raise (
#                     Exception(
#                         "structural scan for channel {} on scope {} not implemented!".format(
#                             lightsource, scope
#                         )
#                     )
#                 )
#             elif scope == "innovation core spinning disk":
#                 mmc.setConfig("Channel", "405")
#                 mmc.setProperty("DAC405", "Volts", 2)
#                 mmc.setExposure(100)

#         elif lightsource == "561-700-75m":
#             if scope == "torstoscope spinning disk":
#                 raise (
#                     Exception(
#                         "structural scan for channel {} on scope {} not implemented!".format(
#                             lightsource, scope
#                         )
#                     )
#                 )
#             elif scope == "innovation core spinning disk":
#                 mmc.setConfig("Channel", "561-700-75m")
#                 mmc.setProperty("DAC561", "Volts", 0.4)
#                 mmc.setExposure(50)

#         elif lightsource == "488-605-70m":
#             if scope == "torstoscope spinning disk":
#                 raise (
#                     Exception(
#                         "structural scan for channel {} on scope {} not implemented!".format(
#                             lightsource, scope
#                         )
#                     )
#                 )
#             elif scope == "innovation core spinning disk":
#                 mmc.setConfig("Channel", "488-605-70m")
#                 mmc.setProperty("DAC488", "Volts", 0.5)
#                 mmc.setExposure(50)

#         else:
#             raise (
#                 Exception(
#                     "Lightsource {} for MMSubroutine structural scan not recognized!".format(
#                         lightsource
#                     )
#                 )
#             )

#         # default is 10x scans
#         if num_images is None:
#             num_images = zsize * 10

#         imgs = snapScan(mmc, num_images, args)
#         saveScanTiffs(fname_root + "_{}.tiff".format(lightsource), imgs)

#         if scope == "torstoscope spinning disk":
#             if lightsource == "Brightfield":
#                 mmc.setConfig("DIA", "OFF")

#     except Exception as err:
#         logging.exception(
#             "Error while trying to take structural scan. No scan taken: {}. NOT saved!".format(
#                 err
#             )
#         )


# def snapScan(mmc, num_images, args):
#     """Takes num_images images as snaps"""

#     # if using pycromanager we need height/width from mmc
#     # but how do we get pycromanager info to trickle down to this function?
#     # scope = args['gooey_args']['microscope_name']
#     backend = args["gooey_args"]["acquisition_backend"]
#     if backend == "pycromanager":
#         xsize = args["roi"][2]
#         ysize = args["roi"][3]

#     imglist = []
#     for i in range(num_images):

#         mmc.snapImage()

#         if backend == "pycromanager":
#             img = mmc.getImage().astype(np.uint16).reshape((ysize, xsize))
#         else:
#             img = mmc.getImage().astype(np.uint16)

#         imglist.append(img)

#     return np.array(imglist)


def saveScanTiffs(fname, img_array):
    """Stub: TIFF saving removed."""
    logging.debug("saveScanTiffs: stub, no-op")

#         logging.critical(
#             "Trying to salvage recording! Writing to {}".format(savefilename)
#         )
#         tf.imwrite(fname, img_array)


# def load_structural_images(structural_scan_dir, dt):
#     """Load structural images"""

#     # get root for all structural tiffs
#     structural_scan_root = structural_scan_dir + "\\{}".format(dt)
#     logging.info("Loading structural scans from {}".format(structural_scan_root))

#     # load different tiffs
#     ls = ["488", "561", "Brightfield"]
#     scans = []

#     # load images
#     for l in ls:
#         d = tf.imread(structural_scan_root + "_{}.tiff".format(l))
#         scans.append(d)

#     # convert to array
#     arr = np.array(scans)
#     return arr
