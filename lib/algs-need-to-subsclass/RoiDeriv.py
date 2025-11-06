from lib import ImageProcessor
import numpy as np
import scipy.ndimage
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import logging
#from image_registration import cross_correlation_shifts
from lib import wbliveUtils as utils
from lib import MMSubroutines

logging.basicConfig(level=logging.DEBUG)


sns.set_style("darkgrid")


class RoiDeriv:
    def __init__(self, args):

        # general params
        self.args = args
        # self.structural_scan_dir = args["structural_scan_dir"]
        self.rec_id = self.args["id"]
        self.roi = self.args["roi"]

        # params about stimulation
        self.args = args
        self.frames_to_grab = self.args["gooey_args"]["total_frames"]
        self.zsize = self.args["gooey_args"]["zsize"]
        self.stim_threshold_pos = self.args["gooey_args"]["stim_threshold_pos"]
        self.stim_threshold_neg = self.args["gooey_args"]["stim_threshold_neg"]
        self.skip_stimulation_probability = self.args["gooey_args"][
            "skip_stimulation_probability"
        ]
        self.delay_stimulation_probability = self.args["gooey_args"][
            "delay_stimulation_probability"
        ]
        self.stim_intensity_ops = self.args["gooey_args"]["stim_intensity_options"]
        self.frames_to_stimulate_for_options = self.args["gooey_args"][
            "frames_to_stimulate_for_options"
        ]
        self.stim_delay_frames_ops = self.args["gooey_args"][
            "stim_delay_frames_options"
        ]
        self.frames_to_cool_down_after_stim = self.args["gooey_args"]["stim_cooldown"]

        # initialize internal holder vars
        zsize = self.zsize
        foo, bar, xsize, ysize = self.args["roi"]
        self.current_volume = np.zeros((zsize, ysize, xsize))
        self.masked_pix_list = []
        self.globmo = []

        # params for quantification
        self.numpix = 0
        self.deriv_avg_window = 3

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

        # get a roi from user
        print("Please enter ROI for the RoiDeriv Alg. Press 'u' to upload when done.")
        pgons = utils.draw_polygons_on_structural_image(
            datadir=self.structural_scan_dir,
            rec_id=self.rec_id,
            use_mip=False,
            zsize=self.zsize,
            finfo="-RoiDeriv",
        )
        self.user_submitted_polygons = pgons

        # save mip at timepoint zero for moco
        data = MMSubroutines.load_structural_images(
            self.structural_scan_dir, self.rec_id
        )
        vol = data[0, :, :, :]
        self.t0_mip = vol.max(axis=0)

        # calculate initial mask
        self.current_mask = utils.polygons_to_masks(
            self.user_submitted_polygons,
            xoffset=0,
            yoffset=0,
            convert_imagespace_to_polygonspace=False,
            frame_dims=self.current_volume.shape,
        )

        # approximate how many pixels we'll be taking from masked regions
        pix_arr = vol[np.nonzero(self.current_mask)]
        self.numpix = int(0.3 * len(pix_arr))

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
            # some important metadata specific to this alg
            "user_submitted_polygons": self.user_submitted_polygons,
            "t0_mask": utils.lazy_serialize(self.t0_mip),
            "numpix": self.numpix,
            "globmo": self.globmo,
        }

        return metadata

    # we often have a period at the beginning of the expr where we need to collect more data before we can start doing things
    def skip(self):

        # just do nothing, mainly for preserving indexing
        self.live_data.append(0)
        self.raw_signal_list.append(0)
        self.globmo.append([0, 0])

    def process_frame(self, img, zndx):

        # start timer
        process_tstart = time.time()

        # apply mask to frame and get a list of pixels
        pix = img[np.nonzero(self.current_mask[zndx])]
        self.masked_pix_list.append(pix)

        # add to running list of current volume
        self.current_volume[zndx, :, :] = img

        # end and append timer
        process_tend = time.time()
        self.process_time_list.append(process_tend - process_tstart)

    def process_volume(self):

        # start timer
        model_tstart = time.time()

        # aggregate pixel lists from masks applied to each z
        pix_arr = np.array(self.masked_pix_list)
        pix_arr.sort()
        self.masked_pix_list = []

        # take avg of n brightest pixels from source
        raw = np.mean(pix_arr[-self.numpix :])
        self.raw_signal_list.append(raw)

        # smoothen and store deriv
        deriv = np.mean(self.raw_signal_list[-self.deriv_avg_window :])
        self.live_data.append(deriv)

        # calculate global motion and correct polygons for next round's masks
        mip = self.current_volume.max(axis=0)
        xoff, yoff = self.get_globmo(self.t0_mip, mip)
        self.current_mask = utils.polygons_to_masks(
            self.user_submitted_polygons,
            xoffset=xoff,
            yoffset=yoff,
            convert_imagespace_to_polygonspace=False,
            frame_dims=self.current_volume.shape,
        )

        # store globmo
        self.globmo.append([xoff, yoff])

        # end and append timer
        model_tend = time.time()
        self.model_time_list.append(model_tend - model_tstart)

    def check_stim(self, image_ndx, cooldown_counter):

        # to keep consistent with outter loop
        img_count = image_ndx + 1

        # grab current measurement
        current_state = self.live_data[-1]

        # initialize output var
        stim_params = {}
        new_cooldown = cooldown_counter

        # check if we want to stim
        if current_state > self.stim_threshold_pos and not cooldown_counter:

            logging.debug("Rise event detected!")

            # with some probability, don't stimulate!
            myrand = random.random()
            if myrand < self.skip_stimulation_probability:
                self.skipped_stimulation_list.append(image_ndx)
                new_cooldown = self.frames_to_cool_down_after_stim
            else:

                # get stim params from randomly initialized lists
                stim_intensity = random.choice(self.stim_intensity_ops)
                num_stim_frames = random.choice(self.frames_to_stimulate_for_options)

                # random to decide whether to stim right away or wait a little
                myrand = random.random()
                if myrand < self.delay_stimulation_probability:

                    # get how much we want to delay
                    delay = random.choice(self.stim_delay_frames_ops)
                    self.delayed_stimulation_list.append(img_count + delay)

                    # add stim to count
                    stim_params["stim_on"] = img_count + delay
                    stim_params["stim_off"] = img_count + delay + num_stim_frames
                    stim_params["stim_marker"] = "high"
                    stim_params["stim_intensity"] = stim_intensity
                    stim_params["delayed"] = True

                else:

                    # add stim to count
                    stim_params["stim_on"] = img_count
                    stim_params["stim_off"] = img_count + num_stim_frames
                    stim_params["stim_marker"] = "rise"
                    stim_params["stim_intensity"] = stim_intensity

        elif current_state < -self.stim_threshold_neg and not cooldown_counter:

            logging.debug("Fall event detected!")

            myrand = random.random()

            # get stim params from randomly initialized lists
            stim_intensity = random.choice(self.stim_intensity_ops)
            num_stim_frames = random.choice(self.frames_to_stimulate_for_options)

            # check negative trigger
            if myrand < self.delay_stimulation_probability:

                # get how much we want to delay
                delay = random.choice(self.stim_delay_frames_ops)
                self.delayed_stimulation_list.append(img_count + delay)

                # add stim delay
                stim_params["stim_on"] = img_count + delay
                stim_params["stim_off"] = img_count + delay + num_stim_frames
                stim_params["stim_marker"] = "low"
                stim_params["stim_intensity"] = stim_intensity
                stim_params["delayed"] = True

            else:

                # add stim
                stim_params["stim_on"] = img_count
                stim_params["stim_off"] = img_count + num_stim_frames
                stim_params["stim_marker"] = "fall"
                stim_params["stim_intensity"] = stim_intensity

        # if we had a stimulation
        if stim_params:

            # set new cooldown timeout
            new_cooldown = self.frames_to_cool_down_after_stim

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

    def get_globmo(self, f0, f1):

        xoff, yoff = cross_correlation_shifts(f1, f0)

        return xoff, yoff

    def close(self):
        pass

    def plot_model(self, show_plot=False, savefilename=None):
        """Plot current state of model"""

        if len(self.raw_signal_list) < 1:
            logging.warning("There is no alg model to plot!")
            return

        # save stim run figure
        try:

            # make figure
            plt.figure(figsize=(16, 4))

            # plot signal
            to_plot = np.array(self.raw_signal_list).T
            to_plot = scipy.ndimage.gaussian_filter(to_plot, sigma=3)
            to_plot = to_plot - np.min(to_plot, axis=0)
            to_plot = np.max(to_plot, axis=1)
            # xvals = np.linspace(0, 600, num=to_plot.shape[0])
            plt.plot(to_plot)

            # overlay live data
            yoffset = -1
            to_plot = np.array(self.live_data)
            # to_plot = (to_plot - np.amin(to_plot)) / (np.amax(to_plot) - np.amin(to_plot))
            plt.plot(to_plot + yoffset)

            # plot markers for stims (translated to volume instead of frames)
            # res = np.array(stim_on_list) // zsize
            yval = 0
            # plt.scatter(res, yval, marker='^', color='r')

            for ndx, sp in enumerate(self.stim_param_list):

                s_frames = sp["stim_on"]
                sm = sp["stim_marker"]

                # translate to volume index
                s_volumes = s_frames // self.zsize

                if sm == "low":
                    m = "d"
                    c = "b"
                elif sm == "rise":
                    m = "^"
                    c = "r"
                elif sm == "high":
                    m = "o"
                    c = "g"
                elif sm == "fall":
                    m = "v"
                    c = "yellow"
                else:
                    logging.warning("Marker not recognized!")
                    m = "s"
                    c = "k"

                # plt markers
                plt.scatter(s_volumes, yval, marker=m, color=c)

            # plot markers for skipped stims
            res = np.array(self.skipped_stimulation_list) // self.zsize
            yvals = np.zeros(res.shape)
            plt.scatter(res, yvals, marker="s", color="k")

            # decorate
            plt.xlabel("frame")
            plt.ylabel("arbitrary")

            # save fig
            if savefilename:
                plt.savefig(savefilename, bbox_inches="tight")

            # show plot
            if show_plot:
                plt.show()

        except Exception as err:
            print("Exception during stim summary figure generation: {}".format(err))

        finally:
            print("Done saving stim summary figure")
