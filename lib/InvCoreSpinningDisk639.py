import logging
import numpy as np
from lib import StimBaseClass

logging.basicConfig(level=logging.WARNING)


class InvCoreSpinningDisk639(StimBaseClass.StimBaseClass):

    def __init__(self, args, local_handles={}):

        # superconstructor
        super().__init__(args, local_handles=local_handles)

        # load handle to microscope hardware
        self.mmc = self.get_mmc()

        # turn off laser and set shutter to open
        self.MAX_VOLTS = 3.5
        self.spool()


    def spool(self):
        self.mmc.setProperty("DAC639", "Volts", 0)
        self.mmc.setProperty("TTL1-8", "TTL-4", 1)


    def submit_stim_params(self, stim_params, image_ndx):
        if stim_params:

            logging.info('InvCoreSpinningDisk639::submit_stim_params> received stim params: {} image_ndx: {}'.format(stim_params, image_ndx))

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
                if stim_type == 'stream-widefield':
                    self.process_stream_widefield_event(stim_params, image_ndx)

                else:
                    logging.critical('submit_stim_params> stim type {} not recognized!'.format(stim_type))

    
    def process_stream_widefield_event(self, stim_params, image_ndx):

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
            raise(Exception('streamed and dynamically updating stimuli not yet implemented!'))

    def activate_stim(self, stim_intensity=10):

        # stim intensity interpreted as a percent of max - express in volts here
        stim_intensity_volts = np.round(stim_intensity/100*self.MAX_VOLTS, decimals=2)

        # set voltage
        self.mmc.setProperty("DAC639", "Volts", stim_intensity_volts)
        

    def inactivate_stim(self):
        self.mmc.setProperty("DAC639", "Volts", 0)