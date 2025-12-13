import logging, time
from abc import ABC, abstractmethod


class StimBaseClass(ABC):
    """ baseclass for interfacing with stimulus apparatus for wb-live acquisition """

    def __init__(self, args, **kwargs):
        """ init stim base class """

        # unpack some housekeeping args
        self.args = args
        self.local_handles = kwargs.get('local_handles', {})
        self.rec_id = self.args["id"]
        self.roi = self.args["roi"]
        self.savedir = self.args["saveroot"]  # ctains folder + / + yyyymmdd-hh-mm-ss
        

        # grab params derived from input gooey
        self.zsize = self.args["gooey_args"]["zsize"]
        self.stim_interface = self.args["gooey_args"]["stim_interface"]
        self.acquisition_backend = args["gooey_args"]["acquisition_backend"]
        self.trigger_alg = self.args["gooey_args"]["trigger_algorithm"]
        self.microscope_name = self.args["gooey_args"]["microscope_name"]

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


    @abstractmethod
    def submit_stim_params(self, stim_params, image_ndx):
        """ this function is called when stimulus parameters are submitted to the stimulus interface """
        pass


    @abstractmethod
    def activate_stim(self):
        pass


    @abstractmethod
    def inactivate_stim(self):
        pass


    def spool(self):
        """ some algorithms need additional runtime initialization, called spooling here """
        pass


    def check_stim(self, img_count):
        """ external function to identify if current frame is in internal holder for driving/halting stimulation """

        to_activate_stimulation = img_count in self.stim_on_list
        if to_activate_stimulation:

            logging.info(
                "StimBaseClass::check_stim> activating {} stim on frame {}".format(
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


    def get_mmc(self):
        """ return handle to internal micromanager object """

        self.mmc = self.local_handles.get('mmc', None)
        
        if not self.mmc:
            logging.critical(
                "Warning: No MMC oject submitted to stimulus interface."
            )
        
        return self.mmc


    def get_metadata(self, args={}):
        """ return metadata for this stimulus interface """

        # for grabbing stim onset relative to recording start
        t0 = self.args["t0"]
        stim_onset_times_simple = self.get_stim_time_onsets(t0=t0)

        # add to metadata attributes common to all stim interfaces
        metadata = {
            "stim_onset_times_simple": stim_onset_times_simple,
            "stim_on_list": self.stim_on_list,
            "stim_off_list": self.stim_off_list,
            "stim_on_time_list": self.stim_on_time_list,
            "stim_off_time_list": self.stim_off_time_list,
            "stim_param_list": self.stim_param_list,
            
            # timers
            # "submit_stim_params_time_list": self.submit_stim_params_time_list,
            # "process_stim_params_event_time_list": self.process_stim_params_event_time_list,
        }

        return metadata
    

    def close(self):
        """ function to close stimulus interface """
        pass


    def get_stim_time_onsets(self, t0=None):
        """ convenience function to return stimulus onset timing in seconds e.g. for wb-matlab gridplot """

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
    
    @staticmethod
    def initialize_stim_interface(args, local_handles={}):
        """ function to create the relevant object for interacting with hardware depending on gooey and microscope context """

        # get microscope and software backend
        stim_interface = args["gooey_args"]["stim_interface"]
        acquisition_backend = args["gooey_args"]["acquisition_backend"]

        # return a valid stim interface, subclassing to take care of implementation details
        # on different systems
        if stim_interface == "InvCore-LDI-Polygon-640" and acquisition_backend == 'pycromanager':
            from lib import InvCoreLDIPolygon
            stim = InvCoreLDIPolygon.InvCoreLDIPolygon(args, local_handles=local_handles)
        elif stim_interface == "InvCore-ThunderscopeLED3" and acquisition_backend == 'pycromanager':
            from lib import InvCoreThunderscopeLED3
            stim = InvCoreThunderscopeLED3.InvCoreThunderscopeLED3(args, local_handles=local_handles)
        elif stim_interface == "no stim" or stim_interface == 'test' or stim_interface == 'dummy':
            from lib import DummyStim
            stim = DummyStim.DummyStim(args, local_handles=local_handles)
        elif stim_interface == "InvCore-SpinningDisk-639":
            from lib import InvCoreSpinningDisk639
            stim = InvCoreSpinningDisk639.InvCoreSpinningDisk639(args, local_handles=local_handles)
        else:
            raise(Exception('StimBaseClass> alternative stim interfaces not yet tested!'))
        
        return stim