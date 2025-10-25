from lib import StimBaseClass
from lib import wbliveUtils as utils
import numpy as np
import time, logging, json
# import wbliveUtils as utils
logging.basicConfig(level=logging.WARNING)


class InvCoreLDIPolygon(StimBaseClass.StimBaseClass):
    def __init__(self, args, local_handles={}):
        
        # call superconstructor
        super().__init__(args, local_handles=local_handles)

        # load handle to microscope hardware
        self.mmc = self.get_mmc()

        # load slm
        self.slm = self.mmc.getSLMDevice()
        self.mmc.setSLMDevice(self.slm)
        self.DSI_IMGWIDTH = self.mmc.getSLMWidth(self.slm)
        self.DSI_IMGHEIGHT = self.mmc.getSLMHeight(self.slm)
        self.polygon_dims = [self.DSI_IMGWIDTH, self.DSI_IMGHEIGHT]

        # set 600 short pass mirror for simultaneous imaging/ldi light integration
        if self.acquisition_backend == "micromanager pymmcore":
            self.mmc.setConfig("Mightex-Setup", "640-SP")

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

        # this stim inteface we're going to be updating dmd, initiate precompile
        logging.info("Using dynamic mask, intializing Polygon to blank mask.")
        self.user_submitted_mask = np.zeros(shape=self.polygon_dims)
        self.spool()


    def spool(self):
        """ overriding baseclass spool method """ 
                # spool jitted functions with dummy routine
        tstart = time.time()

        # spool jitted functions with dummy routine
        if self.trigger_alg == "PointAndClick" or self.trigger_alg == "HammerOfDawn":
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

        if self.trigger_alg == 'Brainalyzer':

            # pick some stims to make
            ix_arr = np.array([600, 7000, 800, 900])
            iy_arr = np.array([100, 200, 300, 400])
            width_arr = np.array([20, 50, 20, 50])
            height_arr = np.array([20, 30, 40, 50])
            trash = utils.generate_pg_multi_rectangle_mask(
                ix_arr,
                iy_arr,
                width_arr,
                height_arr,
                self.pcx,
                self.pcy,
                self.icx,
                self.icy,
                self.roi[0],
                self.roi[1],
                self.DSI_IMGWIDTH,
                self.DSI_IMGHEIGHT,
            )

        tend = time.time()
        logging.info(
            "Spooling wbliveStimClass functions took {}s".format(tend - tstart)
        )

    def get_metadata(self, args):

        # return base class metadata
        metadata = super().get_metadata(args)
        logging.critical('GOTTA TEST METADATA INHERITANCE: {}'.format(metadata))

        # add more attributes
        metadata["stim_intensity_list"] = self.stim_intensity_list
        metadata["calibration_points"] = self.calibration_points

        return metadata

    #########
    def submit_stim_params(self, stim_params, image_ndx):
        
        # stim params could be nonetype 
        if stim_params:

            logging.info('InvCoreLDIPolygon::submit_stim_params> received stim params: {} image_ndx: {}'.format(stim_params, image_ndx))

            # here if we receive a new stimulus ROI, convert it to a mask and upload that mask to Polygon
            stim_params_keys = stim_params.keys()
            if "event" in stim_params_keys:
                self.update_polygon_mask(stim_params)
            
            # handle different events differently
            stim_type = stim_params['event']['event_type'] 

            # stim event populates internal holders, what exactly is determined by stim type
            self.stim_intensity_list.append(stim_params['event']["stim_intensity"])

            # check if stimulus is pulsed (has both on and off times specified)
            is_dynamic_event = False
            if stim_params.get("stim_on", None) and stim_params.get("stim_off", None):

                # store event objects
                self.stim_param_list.append(stim_params)           
                self.stim_on_list.append(stim_params["stim_on"])
                self.stim_off_list.append(stim_params["stim_off"])

            # if event has incomplete information (e.g. streamed stimulus), flip flag denoting dynamic event and handle accordingly
            else:
                is_dynamic_event = True

                # storing data from streamed stimulus requires updating previous event objects, so handle slightly differently
                if stim_type == 'hammer-of-dawn':

                    # 20240407 do we have to add self.stim_param_list.append(stim_params) here?           
                    self.process_hammer_of_dawn_event(stim_params, image_ndx)

                elif stim_type == 'pulse-rect-roi-list' or stim_type == 'stream-rect-roi-list':

                    # raise(Exception('multi-rect streamed and dynamically updating stimuli not yet implemented!'))
                    self.process_stream_rect_roi_list_event(stim_params, image_ndx)
                else:
                    logging.critical('submit_stim_params> stim type {} not recognized!'.format(stim_type))


    def process_stream_rect_roi_list_event(self, stim_params, image_ndx):

        # in hammer of dawn events we populate list with t (xy) datapoints representing center coordinates of stim over time 
        # in multi-rect streamed events, we already have r (xy) datapoints representing r rois within single point, so we would need different scheme 
        


        # received on but no off
        if stim_params.get('stim_on', None) and not stim_params.get('stim_off', None):

            # add to stim param list 
            self.stim_param_list.append(stim_params)
            self.stim_on_list.append(stim_params["stim_on"])

        # received off but no on
        elif not stim_params.get('stim_on', None) and stim_params.get('stim_off', None):

            # append stim off to internal holder
            self.stim_param_list[-1]["stim_off"] = stim_params["stim_off"]
            self.stim_off_list.append(stim_params["stim_off"])

        # received neither on nor off
        elif not stim_params.get('stim_on', None) and not stim_params.get('stim_off', None):
            
            # this sohuldn't happen unless we're updating xy localization of rois during a running model... 
            raise(Exception('multi-rect streamed and dynamically updating stimuli not yet implemented!'))
        


    def process_hammer_of_dawn_event(self, stim_params, image_ndx):

        cx = int(stim_params["event"]["x"])
        cy = int(stim_params["event"]["y"])
        
        # adjusting so next frame is first frame with set ndx, but keeps same indexing as stim on/off
        t = image_ndx + 1

        # some events don't have all info when submitted e.g. on but no corresponding off, or off but no corresponding on
        # in this case we're modifying e.g. stimulus roi, but not whether stimulus is on, so handle accordingly
        # here we get a stim_on but no stim_off
        if stim_params.get("stim_on", None) and not stim_params.get('stim_off', None):

            # cast event xy to list
            stim_params["event"]["x"] = [cx]
            stim_params["event"]["y"] = [cy]
            stim_params["event"]["dynamic_event_frame_ndx"] = [t]

            # add stim on and intensity to internal holder
            self.stim_on_list.append(stim_params["stim_on"])
            self.stim_intensity_list.append(stim_params["stim_intensity"]) # note location of stim_intensity may have shifted between dev of hammer-of-dawn and brainalyzer
            self.stim_param_list.append(stim_params)

        # here we get a stim_off but no stim_on
        elif not stim_params.get('stim_on', None) and stim_params.get('stim_off', None):

            # append stim off to internal holder
            self.stim_off_list.append(stim_params["stim_off"])

            # append previous stim param
            # don't really need to update mask here but might as well
            self.stim_param_list[-1]["event"]["x"].append(cx)
            self.stim_param_list[-1]["event"]["y"].append(cy)
            self.stim_param_list[-1]["event"][
                "dynamic_event_frame_ndx"
            ].append(t)

        # here we get neither stim_on nor stim_off - just an updated HOD xy
        elif not stim_params.get('stim_on', None) and not stim_params.get('stim_off', None):

            self.stim_param_list[-1]["event"]["x"].append(cx)
            self.stim_param_list[-1]["event"]["y"].append(cy)
            self.stim_param_list[-1]["event"][
                "dynamic_event_frame_ndx"
            ].append(t)


    def update_polygon_mask(self, stim_params):

        # unpack event
        et = stim_params['event']['event_type']
        logging.info("InvCoreLDIPolygon::update_polygon_mask> received {} event: {}".format(et, stim_params))

        # circle-rendering events
        if et == 'circle-click' or et == 'circle-button' or et =='hammer-of-dawn':

            # build polygon mask params
            cx = int(stim_params["event"]["x"])
            cy = int(stim_params["event"]["y"])
            diameter = int(stim_params["event"]["stim_diameter"])

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
                self.DSI_IMGHEIGHT)

            # update mask
            mask = mask * 255  # mask is uint8, but values 1-255 are valid and specify dithering
            self.mmc.setSLMImage(self.slm, mask.astype(np.uint8).flatten())

        elif et == 'pulse-rect-roi-list' or et ==  'stream-rect-roi-list':

            # unpack event
            stim_rect_roi_list = stim_params['event']['stim_rect_roi_list']
            x_list = stim_rect_roi_list['x']
            y_list = stim_rect_roi_list['y']
            width_list = stim_rect_roi_list['width']
            height_list = stim_rect_roi_list['height']

            # build polygon mask params
            mask = utils.generate_pg_multi_rectangle_mask(np.array(x_list),np.array(y_list),np.array(width_list),np.array(height_list),self.pcx,self.pcy,self.icx,self.icy,self.roi[0],self.roi[1],self.DSI_IMGWIDTH,self.DSI_IMGHEIGHT)

            # update mask
            mask = mask * 255  # mask is uint8, but values 1-255 are valid and specify dithering
            self.mmc.setSLMImage(self.slm, mask.astype(np.uint8).flatten())
            
        elif et == 'full-field-button':

            # as of like, 2022 this was the best way of updating mask
            self.mmc.setSLMPixelsTo(self.slm, 255)


    #########

    def activate_stim(self, stim_intensity=10):
        self.mmc.setProperty("89 North Laser Diode Illuminator", "640 Intensity", int(stim_intensity))

    def inactivate_stim(self):
        self.mmc.setProperty("89 North Laser Diode Illuminator", "640 Intensity", 0)

    def get_Polygon_calibration_points(self, calibration_fname="./res/peripherals/Mightex Polygon P1000/calibrations.json"):
        """ load calibraiton points for polygon saved in separate json file """
    
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
            logging.error("Error while trying to load calibration points: {}".format(err))
            raise (err)

        return pcx, pcy, icx, icy


