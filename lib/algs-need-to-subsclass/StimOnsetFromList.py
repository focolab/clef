from lib import ImageProcessor
import numpy as np
import scipy.ndimage
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from lib import wbliveUtils as utils
from lib import MMSubroutines

logging.basicConfig(level=logging.DEBUG)


sns.set_style("darkgrid")


class StimOnsetFromList:
    def __init__(self, args):

        # general alg params
        self.args = args
        self.rec_id = self.args["id"]
        self.roi = self.args["roi"]
        self.frames_to_grab = self.args["gooey_args"]["total_frames"]
        self.zsize = self.args["gooey_args"]["zsize"]

        # handle user-provided alg params. should clean up and do better because these may be ignored based on specifics of alg
        self.stim_intensity_ops = self.args["gooey_args"]["stim_intensity_options"]
        self.frames_to_stimulate_for_options = self.args["gooey_args"][
            "frames_to_stimulate_for_options"
        ]

        # alg-specific params
        self.user_provided_stim_onsets = self.args["gooey_args"][
            "stim_onset_list_options"
        ]

        # algs might want to generate semantic warnings based on user input to flag user to potentially undesired behavior
        self.parse_alg_semantic_warnings()

        # output holder vars
        self.live_data = []
        self.raw_signal_list = []
        self.skipped_stimulation_list = []
        self.delayed_stimulation_list = []
        self.process_time_list = []
        self.model_time_list = []
        self.stim_param_list = []

    # whatever setup needs to be done with model
    def initialize_model(self):

        fname_root = self.args["id"]
        print("Initializing model with {}".format(fname_root))

        # for reproducible psudorandomness
        random.seed(fname_root)

    # function to grab algorithm arguments, spitting out semantic warnings (i.e. based on input arguments, may lead to different behavior than what the user expects)
    def parse_alg_semantic_warnings(self):

        # if provided stimulus is in baseline window, we don't do stimulation
        # so flag user that that's the case
        frames_baseline_window = self.args["gooey_args"]["rec_baseline"]
        for s in self.user_provided_stim_onsets:
            if s < frames_baseline_window:
                logging.warning(
                    "Warning, user-provided stimulus at frame {} is in the baseline period of {}, so no stim will be triggered.".format(
                        s, frames_baseline_window
                    )
                )

    def get_metadata(self, args=None):
        """Get metadata which we might want to save for later e.g. for
        writing to file"""

        metadata = {
            "live_data": self.live_data,
            "raw_signal": self.raw_signal_list,
            "skipped_stimulation_list": self.skipped_stimulation_list,
            "process_time_list": self.process_time_list,
            "model_time_list": self.model_time_list,
            "stim_param_list": self.stim_param_list,
            "delayed_stimulation_list": self.delayed_stimulation_list,
            # important metadata specific to this alg
            "user_provided_stim_onsets": self.user_provided_stim_onsets,
        }

        return metadata

    # we often have a period at the beginning of the expr where we need to collect more data before we can start doing things
    def skip(self):

        # just do nothing, preserve indexing or intermediate variable states
        self.live_data.append(0)
        self.raw_signal_list.append(0)

    def process_frame(self, img, zndx):

        # start timer
        process_tstart = time.time()

        # normally we process individual frames here

        # end and append timer
        process_tend = time.time()
        self.process_time_list.append(process_tend - process_tstart)

    def process_volume(self):

        # start timer
        model_tstart = time.time()

        # just place some zeros for continuity
        self.raw_signal_list.append(0)
        self.live_data.append(0)

        # end and append timer
        model_tend = time.time()
        self.model_time_list.append(model_tend - model_tstart)

    def check_stim(self, image_ndx, cooldown_counter):

        # to keep consistent with outter loop
        img_count = image_ndx + 1

        # grab current measurement
        # current_state = self.live_data[-1]

        # initialize output var
        stim_params = {}
        # new_cooldown = cooldown_counter
        new_cooldown = 0

        # if upcoming frame is when we want to stimulate based on user-provided arguments, say so
        if img_count in self.user_provided_stim_onsets:

            # get stim params from randomly initialized lists
            stim_intensity = random.choice(self.stim_intensity_ops)
            num_stim_frames = random.choice(self.frames_to_stimulate_for_options)

            stim_params["stim_on"] = img_count
            stim_params["stim_off"] = img_count + num_stim_frames
            stim_params["stim_intensity"] = stim_intensity

        # if we had a stimulation
        if stim_params:

            # set new cooldown timeout, in this case we're ignoring
            # new_cooldown = self.frames_to_cool_down_after_stim

            # check if recording is about to end, in which case we don't want to stimulate
            if stim_params["stim_off"] > self.frames_to_grab:

                # if stim was delayed, remove reference
                was_delayed = stim_params.get("delayed", False)
                if was_delayed:
                    self.delayed_stimulation_list.pop(-1)

                # remove stimulus info
                stim_params = {}

            self.stim_param_list.append(stim_params)

        return stim_params, new_cooldown

    def close(self):
        pass

    def plot_model(self, show_plot=False, savefilename=None):
        """Plot current state of model"""

        logging.warning("There is no alg model to plot!")
