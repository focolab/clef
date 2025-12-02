from lib import ImageProcessor
import numpy as np
import scipy.ndimage
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


class DynamicRangeDeriv:
    def __init__(self, args):

        # params
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

        # objects for image processing
        self.ip = ImageProcessor.ImageProcessor()

        # initialize internal holder vars
        self.slopes_zlist = []
        for z in range(self.zsize):
            self.slopes_zlist.append([])

        # output holder vars
        self.live_data = []
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

    def get_metadata(self, args=None):
        """Get metadata which we might want to save for later e.g. for
        writing to file"""

        metadata = {
            "live_data": self.live_data,
            "skipped_stimulation_list": self.skipped_stimulation_list,
            "process_time_list": self.process_time_list,
            "model_time_list": self.model_time_list,
            "stim_param_list": self.stim_param_list,
            "delayed_stimulation_list": self.delayed_stimulation_list,
        }

        return metadata

    def skip(self):
        self.live_data.append(0)

    def process_frame(self, img, zndx):

        # start timer
        process_tstart = time.time()

        # do processing
        pv = process_frame(img, self.ip)
        slope = (pv[-1] - pv[0]) / pv[0]
        self.slopes_zlist[zndx].append(slope)

        # end and append timer
        process_tend = time.time()
        self.process_time_list.append(process_tend - process_tstart)

    def process_volume(self):

        # start timer
        model_tstart = time.time()

        # grab last n measurements
        temp_slopes_zlist = []
        for z in range(self.zsize):
            temp_slopes_zlist.append(self.slopes_zlist[z][-100:])
        model = np.array(temp_slopes_zlist).T

        # smooth
        zmax = process_model(model)

        # convert to hard deriv
        deriv = process_zmax(zmax)

        # store latest point...
        self.live_data.append(deriv[-1])

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

            print("Rise event detected!")

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

            print("Fall event detected!")

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

    def plot_model(self, show_plot=False, savefilename=None):
        """Plot current state of model"""

        if len(self.slopes_zlist[0]) < 1:
            print("There is no alg model to plot!")
            return

        # save stim run figure
        try:

            # make figure
            plt.figure(figsize=(16, 4))

            # plot signal
            to_plot = np.array(self.slopes_zlist).T
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
                    print("Marker not recognized!")
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

    def close(self):

        pass


# for live processing
def process_frame(frame, ip):

    # grab points
    xys_list = ip.segmentchunk(frame.astype(np.float32))

    # if no xys, skip
    if len(xys_list) == 0:
        return np.array([1])

    # grab points
    pix = []
    xys = np.array(xys_list) - 1  # -1 because of single pixel offset bug...
    pix.append(frame[xys[:, 0], xys[:, 1]])
    pix.append(frame[xys[:, 0], xys[:, 1] - 1])
    pix.append(frame[xys[:, 0], xys[:, 1] + 1])

    pix.append(frame[xys[:, 0] - 1, xys[:, 1]])
    pix.append(frame[xys[:, 0] - 1, xys[:, 1] - 1])
    pix.append(frame[xys[:, 0] - 1, xys[:, 1] + 1])

    pix.append(frame[xys[:, 0] + 1, xys[:, 1]])
    pix.append(frame[xys[:, 0] + 1, xys[:, 1] - 1])
    pix.append(frame[xys[:, 0] + 1, xys[:, 1] + 1])

    # flatten and sort cell-averages
    flat = np.sort(np.array(pix).mean(axis=0).flatten())

    return flat


def process_model(model):

    # some filtering
    model = scipy.ndimage.gaussian_filter(model, sigma=3)

    # avg across zs
    model = np.max(model, axis=1)

    return model


def process_zmax(zmax, box_size=8):

    dzmax = []
    for t in range(box_size, zmax.shape[0]):

        # grab and average
        val = np.mean(zmax[t - box_size : t])

        # subtract from last point
        dzmax.append(zmax[t] - val)

    return dzmax
