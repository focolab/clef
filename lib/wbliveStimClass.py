import time
import numpy as np
import socket
import os
import subprocess
import logging
import json
from lib import MMSubroutines
from lib import wbliveUtils as utils

# logging.basicConfig(level=logging.INFO)


class wbliveStimClass:
    """class for interfacing with stimulus apparatus for wb-live acquisition"""

    def __init__(self, args, local_handles={}):

        # general housekeeping variables
        self.args = args
        self.rec_id = self.args["id"]
        self.roi = self.args["roi"]
        self.savedir = self.args["saveroot"]  # ctains folder + / + yyyymmdd-hh-mm-ss
        self.zsize = self.args["gooey_args"]["zsize"]

        # booleans
        self.SAVE_STRUCTURAL_SCAN = self.args["gooey_args"]["save_structural_scan"]
        self.USE_STATIC_STIM_ROI = self.args["gooey_args"]["use_static_stim_roi"]
        if self.USE_STATIC_STIM_ROI:
            logging.warning(
                "Selected static stim ROI, but static stim ROI is deprecated!"
            )
            self.USE_STATIC_STIM_ROI = False

        # conditional initialization
        if self.SAVE_STRUCTURAL_SCAN != "" and self.SAVE_STRUCTURAL_SCAN != "none":
            # this will only be present if a structural scan was conducted, otherwise will throw exception
            self.structural_scan_dir = self.args["structural_scan_dir"]

        # get what type of stim to interface with
        self.stim_interface = self.args["gooey_args"]["stim_interface"]

        # what backend?
        self.acquisition_backend = args["gooey_args"]["acquisition_backend"]

        # if we have stimulus params, put them in holder variables
        self.stim_diameter = int(self.args["gooey_args"]["stimulus_diameter"])

        # holder variables shared between stimulus interfaces
        self.stim_on_list = []
        self.stim_off_list = []
        self.stim_on_time_list = []
        self.stim_off_time_list = []
        self.stim_intensity_list = []
        self.stim_param_list = []

        # timers
        self.submit_stim_params_time_list = []
        self.process_stim_params_event_time_list = []

        # slm stim interfaces, if used with alg where mask can be changed, may have multiple masks
        # in a recording. if so store that here
        self.dynamic_mask_list = []
        IS_DYNAMIC_MASK = False
        trigger_alg = self.args["gooey_args"]["trigger_algorithm"]
        if trigger_alg == "PointAndClick" or trigger_alg == "HammerOfDawn":
            IS_DYNAMIC_MASK = True

        # variables specific to each stim interface
        if self.stim_interface == "no stim":
            pass

        elif self.stim_interface == "LMM5_561":

            # load handle to microscope hardware
            self.mmc = local_handles["mmc"]
            if not self.mmc:
                logging.critical(
                    "Warning: No MMC oject submitted to stimulus interface."
                )

        elif (
            self.stim_interface == "Polygon1000_590"
            or self.stim_interface == "Polygon1000_470"
        ):

            self.DSI_IMGWIDTH = 912
            self.DSI_IMGHEIGHT = 1140
            self.polygon_dims = [self.DSI_IMGWIDTH, self.DSI_IMGHEIGHT]

            # initialize connection to polygon and light source
            polygon_logfile = self.savedir + "_polygon_output.txt"
            bls_logfile = self.savedir + "_bls_output.txt"
            self.polygon_process, self.polygon_socket = initialize_polygon(
                polygon_logfile
            )
            self.bls_process, self.bls_socket = initialize_bls(bls_logfile)

            # get hardcoded calibration points for polygon
            (pcx, pcy, icx, icy) = self.get_Polygon_calibration_points()
            self.calibration_points = {"pcx": pcx, "pcy": pcy, "icx": icx, "icy": icy}
            self.pcx = np.array(pcx)
            self.pcy = np.array(pcy)
            self.icx = np.array(icx)
            self.icy = np.array(icy)

            # initialize a full-field mask
            self.full_field_stim_mask = np.ones(shape=self.polygon_dims)

            # if trying to use stim ROI but no structural scan was taken, default to full-field
            if self.USE_STATIC_STIM_ROI and not self.SAVE_STRUCTURAL_SCAN:
                logging.critical(
                    "Warning, trying to use stimulus ROI but SAVE_STRUCTURAL_SCAN is disabled. Currently ROI requires a structural scan. Defaulting to full-field stimulation"
                )
                self.USE_STATIC_STIM_ROI = False

            if self.USE_STATIC_STIM_ROI:

                # get user submitted mask and polygons, in Polygon coordinates
                mymask, mypgons = self.get_user_ROI_in_Polygon_coords()
                self.user_submitted_mask = mymask
                self.user_submitted_polygons = mypgons

            else:

                # generate a full-field mask
                self.user_submitted_mask = self.full_field_stim_mask
                self.user_submitted_polygons = []

            if IS_DYNAMIC_MASK:
                logging.info("Using dynamic mask, intializing Polygon to blank mask.")
                self.user_submitted_mask = np.zeros(shape=self.polygon_dims)
                self.spool()

            # send static mask to polygon
            send_mask_to_polygon(self.polygon_socket, self.user_submitted_mask)

        elif self.stim_interface == "InvCore-LDI-Polygon-640":

            # load handle to microscope hardware
            self.mmc = local_handles["mmc"]
            if not self.mmc:
                logging.critical(
                    "Warning: No MMC oject submitted to stimulus interface."
                )

            # set 600 short pass mirror for simultaneous imaging/ldi light integration
            if self.acquisition_backend == "micromanager pymmcore":
                self.mmc.setConfig("Mightex-Setup", "640-SP")

            # load slm
            self.slm = self.mmc.getSLMDevice()
            self.mmc.setSLMDevice(self.slm)
            self.DSI_IMGWIDTH = self.mmc.getSLMWidth(self.slm)
            self.DSI_IMGHEIGHT = self.mmc.getSLMHeight(self.slm)
            self.polygon_dims = [self.DSI_IMGWIDTH, self.DSI_IMGHEIGHT]

            # set ldi off but open shutter
            self.mmc.setProperty("89 North Laser Diode Illuminator", "640 Intensity", 0)
            self.mmc.setSLMPixelsTo(self.slm, 0)
            self.mmc.setShutterOpen("89 North Laser Diode Illuminator", True)

            # initialize a full-field mask
            self.full_field_stim_mask = (
                np.ones(shape=self.polygon_dims).astype(np.uint8) * 255
            )

            # get hardcoded calibration points for polygon
            (pcx, pcy, icx, icy) = self.get_Polygon_calibration_points()
            self.calibration_points = {"pcx": pcx, "pcy": pcy, "icx": icx, "icy": icy}
            self.pcx = np.array(pcx)
            self.pcy = np.array(pcy)
            self.icx = np.array(icx)
            self.icy = np.array(icy)

            # if we're going to be updating dmd, initiate precompile
            if IS_DYNAMIC_MASK:
                logging.info("Using dynamic mask, intializing Polygon to blank mask.")
                self.user_submitted_mask = np.zeros(shape=self.polygon_dims)
                self.spool()

        elif self.stim_interface == "solenoid":

            # load handle to microscope hardware
            self.mmc = local_handles["mmc"]
            if not self.mmc:
                logging.critical(
                    "Warning: No MMC oject submitted to stimulus interface."
                )

            # intialize solenoid with constant ontime
            # in milliseconds
            offtime = 500000  # this number doesn't matter because we're not using the arduino to sequence our stims
            ontime = 500000  # this number is effectively "constantly on"
            cmd = "F,{},{}".format(offtime, ontime)
            send_serial_command(cmd, self.mmc)

    # function to precompile e.g. numba functions
    def spool(self):

        # spool jitted functions with dummy routine
        tstart = time.time()

        trash = utils.generate_pg_ellipse_mask(
            self.DSI_IMGWIDTH // 2,
            self.DSI_IMGHEIGHT // 2,
            self.pcx,
            self.pcy,
            self.icx,
            self.icy,
            self.stim_diameter,
            self.roi[0],
            self.roi[1],
            self.DSI_IMGWIDTH,
            self.DSI_IMGHEIGHT,
        )

        tend = time.time()
        logging.info(
            "Spooling wbliveStimClass functions took {}s".format(tend - tstart)
        )

    # function periodically called to check if stim needs to be issued
    def submit_stim_params(self, stim_params, image_ndx):

        # if we have valid stimulus parameters
        if stim_params:

            tstart = time.time()

            # if we don't want stimulation, just return
            # because we never add stimulus parameters to any internal data structures,
            # advancing state with "check stim" will never do anything
            if self.stim_interface == "no stim":
                return

            logging.debug(
                "wbliveStimClass received stimulus params: {}".format(stim_params)
            )

            # store event information in holder variables
            stim_params_keys = stim_params.keys()

            # if event has incomplete information, flip flag
            is_dynamic_event = False
            if stim_params.get("stim_on", None) and stim_params.get("stim_off", None):
                # if "stim_on" in stim_params_keys and "stim_off" in stim_params_keys:
                self.stim_on_list.append(stim_params["stim_on"])
                self.stim_off_list.append(stim_params["stim_off"])
                self.stim_intensity_list.append(stim_params["stim_intensity"])

                # store event object
                self.stim_param_list.append(stim_params)

            else:
                is_dynamic_event = True

            # do stim-interface-specific-things
            if self.stim_interface == "LMM5_561":
                pass

            elif (
                self.stim_interface == "Polygon1000_590"
                or self.stim_interface == "Polygon1000_470"
                or self.stim_interface == "InvCore-LDI-Polygon-640"
            ):

                # here if we receive a new stimulus ROI, convert it to a mask and upload that mask to Polygon
                if "event" in stim_params_keys:
                    self.process_stim_params_event(stim_params)

                # some events don't have all info when submitted
                # on but no corresponding off
                # off but no corresponding on
                # just XY because we're updating position on DMD
                # so handle accordingly
                if (
                    is_dynamic_event
                    and stim_params["event"]["event_type"] == "hammer-of-dawn"
                ):

                    cx = int(stim_params["event"]["x"])
                    cy = int(stim_params["event"]["y"])
                    t = (
                        image_ndx + 1
                    )  # so next frame is first frame with set ndx. keeps same indexing as stim on/off

                    # on but no off, save xy and on frame and add to list
                    if stim_params.get("stim_on", None) and not stim_params.get(
                        "stim_off", None
                    ):
                        # if "stim_on" in stim_params_keys and "stim_off" not in stim_params_keys:

                        # cast event xy to list
                        stim_params["event"]["x"] = [cx]
                        stim_params["event"]["y"] = [cy]
                        stim_params["event"]["dynamic_event_frame_ndx"] = [t]

                        # add stim on and intensity to internal holder
                        self.stim_on_list.append(stim_params["stim_on"])
                        self.stim_intensity_list.append(stim_params["stim_intensity"])

                        # save stim object
                        self.stim_param_list.append(stim_params)

                    # off but no on, save off frame and add to last event
                    elif not stim_params.get("stim_on", None) and stim_params.get(
                        "stim_off", None
                    ):
                        # elif (
                        # "stim_on" not in stim_params_keys
                        # and "stim_off" in stim_params_keys
                        # ):

                        # append stim off to internal holder
                        self.stim_off_list.append(stim_params["stim_off"])

                        # append previous stim param
                        # don't really need to update mask here but might as well
                        self.stim_param_list[-1]["event"]["x"].append(cx)
                        self.stim_param_list[-1]["event"]["y"].append(cy)
                        self.stim_param_list[-1]["event"][
                            "dynamic_event_frame_ndx"
                        ].append(t)

                    # no off or on, is pure xy update
                    elif not stim_params.get("stim_on", None) and not stim_params.get(
                        "stim_off", None
                    ):
                        # elif (
                        # "stim_on" not in stim_params_keys
                        # and "stim_off" not in stim_params_keys
                        # ):

                        self.stim_param_list[-1]["event"]["x"].append(cx)
                        self.stim_param_list[-1]["event"]["y"].append(cy)
                        self.stim_param_list[-1]["event"][
                            "dynamic_event_frame_ndx"
                        ].append(t)

            tend = time.time()
            self.submit_stim_params_time_list.append(tend - tstart)

    def process_stim_params_event(self, stim_params):

        et = stim_params["event"]["event_type"]
        tstart = time.time()
        if et == "circle-click" or et == "circle-button" or et == "hammer-of-dawn":

            # build napari-polygon from cx, cy
            cx = int(stim_params["event"]["x"])
            cy = int(stim_params["event"]["y"])
            diameter = int(stim_params["event"]["stim_diameter"])

            logging.debug(
                "wbliveStimClass received {} event; x={}; y={}; diameter={}".format(
                    et, cx, cy, diameter
                )
            )

            # generate_pg_circle_mask(ix, iy, pcx, pcy, icx, icy, diameter, DSI_IMGHEIGHT, DSI_IMGWIDTH)
            # draw ellipse because of coordinate anisotropy
            mask = utils.generate_pg_ellipse_mask(
                cx,
                cy,
                self.pcx,
                self.pcy,
                self.icx,
                self.icy,
                diameter,
                self.roi[0],
                self.roi[1],
                self.DSI_IMGWIDTH,
                self.DSI_IMGHEIGHT,
            )

            if (
                self.stim_interface == "Polygon1000_590"
                or self.stim_interface == "Polygon1000_470"
            ):

                # update polygon
                send_mask_to_polygon(self.polygon_socket, mask)

            elif self.stim_interface == "InvCore-LDI-Polygon-640":

                # dithering!
                mask = mask * 255
                self.mmc.setSLMImage(self.slm, mask.astype(np.uint8).flatten())

                # set mask to polygon with micromanager slm api
                # self.mmc.loadSLMSequence(self.slm, [mask.tobytes()])
                # self.mmc.startSLMSequence(self.slm)

            # add to list, if hammer of dawn there's too many so don't bother
            if et != "hammer-of-dawn":
                self.dynamic_mask_list.append(mask)

        elif stim_params["event"]["event_type"] == "full-field-button":

            logging.debug("wbliveStimClass received {} event".format(et))

            # generate full field mask
            if (
                self.stim_interface == "Polygon1000_590"
                or self.stim_interface == "Polygon1000_470"
            ):

                send_mask_to_polygon(self.polygon_socket, self.full_field_stim_mask)

            elif self.stim_interface == "InvCore-LDI-Polygon-640":

                # set slm pixels
                self.mmc.setSLMPixelsTo(self.slm, 255)

            self.dynamic_mask_list.append(self.full_field_stim_mask)

        tend = time.time()
        self.process_stim_params_event_time_list.append(tend - tstart)

    # function to get the appropriate calibration points based on microscope config
    def get_Polygon_calibration_points(self):

        # get objective from microscope configuration
        scope = self.args["gooey_args"]["microscope_name"]
        if scope == "torstoscope spinning disk":
            obj = self.args["configs"]["Objective"]
            logging.info(
                "Grabbing Polygon calibration points for objective {}".format(obj)
            )

            # if we're at 40x use this. sometimes we accidentally screw the objective into the wrong hole though
            if obj == "6-Plan Apo 40x NA 1.25 WI":

                # 20210927-20-04-43
                # get center pixel of image in polygon mask
                x, p1y = [632, 255]
                x, p2y = [632, 463]
                x, p3y = [664, 640]

                p1x, y = [317, 476]
                p2x, y = [632, 463]
                p3x, y = [1024, 463]

                # get center pixel of image patterned on the sample, hardcoded
                x, i1y = [464, 154]
                x, i2y = [467, 509]
                x, i3y = [493, 809]

                i1x, y = [200, 531]
                i2x, y = [467, 509]
                i3x, y = [796, 508]

            elif obj == "1-Plan Apo 4x NA 0.20 Dry":

                # 20210604 4x 20210604-17-22-28
                # get center pixel of image in polygon mask
                x, p1y = [568, 274]
                x, p2y = [604, 462]
                x, p3y = [615, 650]

                p1x, y = [297, 439]
                p2x, y = [604, 462]
                p3x, y = [968, 434]

                # get center pixel of image patterned on the sample, hardcoded
                x, i1y = [460, 219]
                x, i2y = [491, 537]
                x, i3y = [499, 853]

                i1x, y = [233, 498]
                i2x, y = [491, 537]
                i3x, y = [795, 488]

            else:
                logging.warning(
                    "Warning while trying to get Polygon calibration points from objective {}, no points found so returning empty lists.".format(
                        obj
                    )
                )
                return [], [], [], []

            # pack individual points
            pcx = [p1x, p2x, p3x]
            pcy = [p1y, p2y, p3y]
            icx = [i1x, i2x, i3x]
            icy = [i1y, i2y, i3y]

        elif scope == "innovation core spinning disk":

            calibration_fname = (
                "./res/peripherals/Mightex Polygon P1000/calibrations.json"
            )

            try:
                with open(calibration_fname) as j:
                    md = json.load(j)
                    calibrations = md["calibrations"]

                    # some logic to get the latest calibration points, could be cleaned up
                    dt_list = []
                    obj = self.mmc.getProperty("ObjectiveTurret", "Label")
                    cam = self.mmc.getCameraDevice()
                    binning = self.mmc.getProperty(cam, "Binning")
                    for cali in calibrations:
                        if cali["objective"] == obj and cali["binning"] == binning:
                            dt = cali["datetime"]
                            dt_list.append(dt)

                    # find latest dt, should throw an error if no matched objective
                    if len(dt_list) > 0:
                        dt_list.sort()
                        latest_dt = dt_list[-1]
                    else:
                        raise (
                            Exception(
                                "No matching calibration found for objective: {}, with binning: {}".format(
                                    obj, binning
                                )
                            )
                        )

                    # grab matching calibration.
                    for cali in calibrations:
                        if cali["objective"] == obj:
                            if cali["datetime"] == latest_dt:
                                pcx = cali["pcx"]
                                pcy = cali["pcy"]
                                icx = cali["icx"]
                                icy = cali["icy"]

            except Exception as err:
                print("Error while trying to load calibration points: {}".format(err))
                raise (err)

        return pcx, pcy, icx, icy

    # function to get user-submitted-roi, which converts from imagespace to polygon space
    def get_user_ROI_in_Polygon_coords(self):

        # get user submitted polygons
        logging.info(
            "Please enter ROI for the Polygon1000. Press 'u' to upload when done."
        )
        pgons = utils.draw_polygons_on_structural_image(
            datadir=self.structural_scan_dir,
            rec_id=self.rec_id,
            use_mip=True,
            zsize=self.zsize,
            finfo="-Polygon1000",
        )

        # convert into mask and transform into appropriate coords system
        mask = utils.polygons_to_masks(
            pgons=pgons,
            convert_imagespace_to_polygonspace=True,
            xoffset=self.roi[0],
            yoffset=self.roi[1],
            frame_dims=self.polygon_dims,
            pcx=self.pcx,
            pcy=self.pcy,
            icx=self.icx,
            icy=self.icy,
        )

        return mask, pgons

    # function to progress stim state, e.g. check if stim needs
    # to be turned on or off
    def check_stim(self, img_count):

        # if this is a frame to do stimulation, switch stim condition
        # turn up 561 laser power

        to_activate_stimulation = img_count in self.stim_on_list
        if to_activate_stimulation:

            logging.info(
                "wbliveStimClass activating {} stim on frame {}".format(
                    self.stim_interface, img_count
                )
            )

            # get stim intensity
            s = self.stim_intensity_list[self.stim_on_list.index(img_count)]
            self.activate_stim(s)
            self.stim_on_time_list.append(time.time())
            # self.stim_on_flag = 1

        # check to turn stim off (e.g. with switching selected lines or with power)
        to_inactivate_stimulation = img_count in self.stim_off_list
        if to_inactivate_stimulation:

            logging.info(
                "wbliveStimClass deactivating {} stim on frame {}".format(
                    self.stim_interface, img_count
                )
            )
            self.inactivate_stim()
            self.stim_off_time_list.append(time.time())
            # self.stim_on_flag = 0

    # function to set the stimulus to a specific intensity
    def activate_stim(self, stim_intensity):

        # set lasers through micro manager
        if self.stim_interface == "LMM5_561":
            self.mmc.setConfig("LMM5-561-intensity", str(stim_intensity))

        # stim for polygon
        elif self.stim_interface == "Polygon1000_590":

            # convert string x% stim intensity to x0.1% int stim intensity
            stim_intensity = int(stim_intensity) * 10
            send_bls_control_command(self.bls_socket, channel=2, value=stim_intensity)

        elif self.stim_interface == "Polygon1000_470":

            # convert string x% stim intensity to x0.1% int stim intensity
            stim_intensity = int(stim_intensity) * 10
            send_bls_control_command(self.bls_socket, channel=1, value=stim_intensity)

        elif self.stim_interface == "solenoid":

            # send serial command (always starts with high voltage on solenoid)
            cmd = "E"
            send_serial_command(cmd, self.mmc)

        elif self.stim_interface == "InvCore-LDI-Polygon-640":

            # set ldi intensity
            try:
                # self.mmc.setShutterOpen("89 North Laser Diode Illuminator", True)
                self.mmc.setProperty(
                    "89 North Laser Diode Illuminator",
                    "640 Intensity",
                    int(stim_intensity),
                )
            except Exception as err:
                logging.error(
                    "Error: {}. Cannot set LDI intensity to {}, defaulting to 10".format(
                        err, stim_intensity
                    )
                )
                self.mmc.setProperty(
                    "89 North Laser Diode Illuminator", "640 Intensity", 10
                )

    # function to inactivate stimulation
    def inactivate_stim(self):
        if self.stim_interface == "LMM5_561":
            self.mmc.setConfig("LMM5-561-intensity", "0")

        elif self.stim_interface == "Polygon1000_590":
            send_bls_control_command(self.bls_socket, channel=2, value=0)

        elif self.stim_interface == "Polygon1000_470":
            send_bls_control_command(self.bls_socket, channel=1, value=0)

        elif self.stim_interface == "solenoid":

            # send serial command to turn off (state when there is no voltage to the solenoid))
            cmd = "D"
            send_serial_command(cmd, self.mmc)

        elif self.stim_interface == "InvCore-LDI-Polygon-640":
            # self.mmc.setShutterOpen("89 North Laser Diode Illuminator", False)
            self.mmc.setProperty("89 North Laser Diode Illuminator", "640 Intensity", 0)

    # function to return all useful information for saving
    def get_metadata(self, args):

        # initialize output variable
        metadata = {}

        # args provided in case we need info for correct metadata output
        t0 = args["t0"]

        # add metadata attributes common to all stim interfaces
        metadata["stim_onset_times_simple"] = self.get_stim_time_onsets(t0=t0)
        metadata["stim_on_list"] = self.stim_on_list
        metadata["stim_off_list"] = self.stim_off_list
        metadata["stim_on_time_list"] = self.stim_on_time_list
        metadata["stim_off_time_list"] = self.stim_off_time_list
        metadata["stim_param_list"] = self.stim_param_list
        metadata["submit_stim_params_time_list"] = self.submit_stim_params_time_list
        metadata[
            "process_stim_params_event_time_list"
        ] = self.process_stim_params_event_time_list

        # stim interface specific metadata
        if self.stim_interface == "LMM5_561":
            pass
        elif (
            self.stim_interface == "Polygon1000_590"
            or self.stim_interface == "Polygon1000_470"
        ):
            metadata["user_submitted_polygons"] = utils.lazy_serialize(
                self.user_submitted_polygons, to_str=True
            )
            metadata["user_submitted_mask"] = utils.lazy_serialize(
                self.user_submitted_mask, to_str=True
            )

            # calibration points
            # metadata["calibration_pts1"] = self.calibration_pts1
            # metadata["calibration_pts2"] = self.calibration_pts2
            metadata["calibration_points"] = self.calibration_points

            # if there were dynamic masks, add one at a time
            masks_serialized = []
            if len(self.dynamic_mask_list) > 0:
                for m in self.dynamic_mask_list:
                    masks_serialized.append(utils.lazy_serialize(m, to_str=True))
            metadata["dynamic_mask_list"] = masks_serialized

        elif self.stim_interface == "InvCore-LDI-Polygon-640":

            # if there were dynamic masks, add one at a time
            masks_serialized = []
            if len(self.dynamic_mask_list) > 0:
                for m in self.dynamic_mask_list:
                    masks_serialized.append(utils.lazy_serialize(m, to_str=True))
            metadata["dynamic_mask_list"] = masks_serialized

        return metadata

    # legacy function to return stimulus onset information e.g. for wb-matlab gridplot
    def get_stim_time_onsets(self, t0=None):

        onsets = self.stim_on_time_list

        # if we provide a starting time, return stim onsets based on that
        # relative time
        if t0:
            new_onsets = []
            for on in onsets:
                on = on - t0
                new_onsets.append(on)
            onsets = new_onsets

        return onsets

    def close(self):

        # take care of any cleaning up that has to be done between sockets etc
        if (
            self.stim_interface == "Polygon1000_590"
            or self.stim_interface == "Polygon1000_470"
        ):

            try:
                logging.debug("Closing polygon and bls subprocesses")
                self.polygon_process.terminate()
                self.bls_process.terminate()
                self.polygon_socket.close()
                self.bls_socket.close()
            except Exception as err:
                logging.exception(
                    "Error while trying to close down stimulus interface: {}".format(
                        err
                    )
                )


# function to initialize polygon connection
def initialize_polygon(
    logfile_fname,
    PORT=5007,
    path_to_exe="C:/Users/confocal/source/repos/polygon-app/x64/Debug/polygon-app.exe",
):

    try:

        # open logfile and pass to subprocess
        f = open(logfile_fname, "w")

        if path_to_exe:
            polygon_process = subprocess.Popen(path_to_exe, stdout=f)
            # self.polygon_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, creationflags=0x08000000)

    except Exception as err:

        logging.exception("Error during launching of polygon app: {}".format(err))
        raise (err)

    # connect to polygon app
    try:

        HOST = "localhost"
        polygon_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        polygon_socket.connect((HOST, PORT))
        logging.debug("Connected to Polygon server app on port {}".format(PORT))

    except Exception as err:
        logging.debug("Error while trying to connect to polygon app: {}".format(err))

    return polygon_process, polygon_socket


# function to initialize connection to bioled light source controller
def initialize_bls(
    logfile_fname,
    PORT=5008,
    path_to_exe="C:/Users/confocal/source/repos/bls-app/x64/Debug/bls-app.exe",
):

    try:

        # open logfile and pass to subprocess
        f = open(logfile_fname, "w")

        if path_to_exe:
            bls_process = subprocess.Popen(path_to_exe, stdout=f)

    except Exception as err:
        logging.exception("Error during launching of bls app: {}".format(err))
        raise (err)

    try:

        HOST = "localhost"
        bls_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bls_socket.connect((HOST, PORT))
        logging.debug("Connected to BLS server app on port {}".format(PORT))

    except Exception as err:
        logging.exception("Error while trying to connect to bls app: {}".format(err))
        raise (err)

    return bls_process, bls_socket


def send_mask_to_polygon(s, mask, preprocessed=False):

    logging.debug("Sending mask of shape {} to polygon.".format(mask.shape))

    if not preprocessed:
        mask = mask.astype(np.bool).T
        mask = np.flip(mask, axis=0)
        mask = np.flip(mask, axis=1)
        mask = np.packbits(mask)

    # send message
    try:

        s.send(mask)

        # grab response and reshape
        # buff_size = 1024
        # resp = s.recv(buff_size)
        # returned = np.frombuffer(res, dtype=np.bool).reshape(msg.shape)

        # print
        # logging.debug("Response: {}".format(resp))

    except Exception as err:
        logging.exception(
            "Exception while trying to send/recv a message to/from the polygon: {}".format(
                err
            )
        )
        raise (err)


# function to send a control signal to bioled light source controller
# modulates channel and brightness in %0.1 units (so 1000 = 100%)
def send_bls_control_command(s, channel, value):

    # write message
    msg = [channel, value]
    msg = np.array(msg)
    msg = msg.tobytes()

    try:
        s.send(msg)

        # grab response and reshape
        # buff_size = 1024
        # resp = s.recv(buff_size)
        # returned = np.frombuffer(res, dtype=np.bool).reshape(msg.shape)

        # print
        # logging.debug("Response from bls: {}".format(resp))

    except Exception as err:
        logging.exception(
            "Exception while trying to send/recv a message to/from the BLS controller: {}".format(
                err
            )
        )
        raise (err)


def send_serial_command(command, mmc, port="COM11"):

    # useful debug output
    logging.debug("Sending serial command: {} to port: {}.".format(command, port))

    endln = "\r"
    mmc.setSerialPortCommand(port, command, endln)
