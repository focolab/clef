import numpy as np

# import napari
import json
import logging
# import scipy.io
import pickle
import platform
from os import path
import codecs
# import numba
# from numba import jit

# conditional import, necessary for standalone testing
try:
    from utils import MMSubroutines
except ImportError as err:
    # change path and try again
    try:
        import MMSubroutines
    except ImportError as err:
        logging.critical('Error while trying to import MMSubroutines: {}'.format(err))

# conditional imports
if platform.system() == "Windows":
    import winsound


logging.basicConfig(level=logging.INFO)


def save_metadata(savefilename, metadata):

    logging.debug("Saving metadata to {}".format(savefilename))

    # save stimulus information
    try:

        with open(savefilename, "w") as outfile:
            json.dump(metadata, outfile, indent=4, sort_keys=True)

    except Exception as err:
        logging.critical("Exception during saving metadata file: {}".format(err))
        logging.debug(f"unsaved metadata: {metadata}")

    finally:
        logging.debug("Done saving acquisition metadata!")


def generate_mip_movie(savefilename, frames, zsize, exposure, quality=6, bitrate="10M", GUI_mode='neural_imaging'):

    logging.debug("Saving MIP movie...")

    # convert to mip movie
    try:

        import imageio

        # grab bounds
        vols_to_grab = frames.shape[0] // zsize
        ysize = frames.shape[1]
        xsize = frames.shape[2]

        # recast array for mip
        tosave = frames.reshape((vols_to_grab, zsize, ysize, xsize)).max(axis=1)

        # grab a frame somewhat in
        # f0 = tosave.flatten()
        # f0.sort()
        # take avg of top % pixelsf0
        # mymax = np.median(f0[-int(xsize * ysize * 0.01) :])
        # mymin = np.amin(f0)

        mymax = np.amax(tosave)
        mymin = np.amin(tosave)

        # compress to MIP, scale to min/max
        tosave[tosave > mymax] = mymax
        # tosave[tosave < mymin] = mymin
        # tosave = (tosave - mymin) / (mymax - mymin) * 255 # how to make this uint16? 
        # tosave = tosave - mymin
        # tosave = (tosave * 255)  // (mymax - mymin) 
        tosave = (tosave * 255) // mymax

        # save Zx speedup (cause that's a nice round number?)
        multiplier = 2
        imageio.mimwrite(
            savefilename + "_{}x.mp4".format(multiplier * zsize),
            tosave.astype(np.uint8),
            fps=multiplier * 1000 // int(exposure),
            # quality=6,
            codec='h264_nvenc',
            output_params=['-b:v', bitrate],
        )

    except Exception as err:
        logging.exception("Exception during MIP movie generation: {}".format(err))


# def prefill_wb_ops(
#     savefileroot,
#     metadata,
#     meta_template_fname="res/meta_template.mat",
#     wboptions_template_fname="res/wboptions_template.mat",
# ):
    
#     import scipy.io

#     # load
#     try:
#         meta = scipy.io.loadmat(meta_template_fname)
#         wbops = scipy.io.loadmat(wboptions_template_fname)
#     except FileNotFoundError as err:
#         logging.exception(
#             "Error while trying to load wb-matlab template param files: {}".format(err)
#         )
#         return

#     try:
#         # grab fields from metadata
#         num_frames = metadata["gooey_args"]["total_frames"]
#         zsize = metadata["gooey_args"]["zsize"]
#         total_time = metadata["frame_time_list"][-1]
#         rec_id = metadata["id"]

#         # prefill form fields eww gross
#         meta["totalTime"][0][0] = total_time

#         meta["fileInfoOverride"][0][0][0][0] = np.array([num_frames // zsize])
#         meta["fileInfoOverride"][0][0][1][0] = np.array([zsize])
#         meta["fileInfoOverride"][0][0][2][0] = np.array([1])

#         if "stim_metadata" in metadata:
#             ons = metadata["stim_metadata"]["stim_onset_times_simple"]
#             meta["stimulus"][0][0][2] = np.array(ons)

#         meta["fileInfo"][0][0][1][0][0] = np.array(rec_id + ".tiff")

#         # x
#         meta["fileInfo"][0][0][2][0][0] = metadata["xsize"]
#         meta["fileInfo"][0][0][6][0][0] = metadata["xsize"]

#         # y
#         meta["fileInfo"][0][0][3][0][0] = metadata["ysize"]
#         meta["fileInfo"][0][0][7][0][0] = metadata["ysize"]

#         # total frames
#         meta["fileInfo"][0][0][4][0][0] = num_frames
#         meta["fileInfo"][0][0][5][0][0] = num_frames

#     except Exception as err:
#         logging.exception(
#             "Error while setting prefilled wb-matlab form fields: {}".format(err)
#         )
#         return

#     # write
#     try:

#         meta_savefilename = savefileroot + "/meta.mat"
#         wbops_savefilename = savefileroot + "/wboptions.mat"
#         scipy.io.savemat(meta_savefilename, meta)
#         scipy.io.savemat(wbops_savefilename, wbops)

#     except Exception as err:
#         logging.exception(
#             "Error while trying to write wb-matlab prefilled files: {}".format(err)
#         )
#         return

def lazy_serialize(res, to_str=True):

    # if we want to convert to a string
    if to_str:

        # serialize into bytes
        b = pickle.dumps(res)

        # encode bytes as base64 bytes
        b64 = codecs.encode(b, "base64")

        # convert to string
        return b64.decode()

    # if we want to take a string and convert it to an obj
    else:

        # take json-loaded string and encode as b64
        b64 = codecs.decode(res.encode(), "base64")

        # load b64 as python obj
        return pickle.loads(b64)


def notify(msg, interface=None, ops={}):

    logging.debug(
        "Notifying wb-live via interface: {}, message: {}".format(interface, msg)
    )

    if interface == "twilio-sms":

        try:

            from twilio.rest import Client

            # Your Account SID from twilio.com/console
            account_sid = ops.get("twilio-sid", "AC4465d6c58ea433aa4917d09617dd8dbc")

            # Your Auth Token from twilio.com/console
            auth_token = ops.get("twilio-auth", "6dfeb1dcf3104b45254c69a05177abf0")
            client = Client(account_sid, auth_token)

            # Numbers
            to_num = ops.get("twilio-to", "+19788442244")
            from_num = ops.get("twilio-from", "+12029337546")

            message = client.messages.create(to=to_num, from_=from_num, body=msg)

            logging.debug("Notify response: {}".format(message.sid))

        except Exception as err:

            logging.error("Error during wbliveUtils::notify, {}".format(err))


def play_wblive_sound(sound_name="laser"):

    # get sound filename
    if sound_name == "laser":
        fname = "./res/audio/laser.wav"
    else:
        logging.error(
            "Error while trying to play sound, sound name {} not recognized".format(
                sound_name,
            )
        )
        return

    # check to make sure file exists
    if not path.exists(fname):
        logging.error(
            "Error while trying to play sound, sound {} not found in {}".format(
                sound_name, fname
            )
        )

        # try one more time from parent directory
        logging.warning("Checking for sound {} in {}".format(sound_name, fname))
        fname = "../" + fname
        if not path.exists(fname):
            logging.error(
                "Error while trying to play sound, sound {} not found in {}".format(
                    sound_name, fname
                )
            )
            return

    # play async sound
    # sound effects are os dependent
    if platform.system() == "Windows":
        winsound.PlaySound(fname, winsound.SND_ASYNC | winsound.SND_ALIAS)

