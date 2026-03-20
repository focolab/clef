import numpy as np
import json
import logging
import pickle
import platform
from os import path
import codecs

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


# def notify(msg, interface=None, ops={}):
#     """Stub: SMS notification removed."""
#     logging.debug("notify: stub, no-op (msg={})".format(msg))


# def play_wblive_sound(sound_name="laser"):

#     # get sound filename
#     if sound_name == "laser":
#         fname = "./res/audio/laser.wav"
#     else:
#         logging.error(
#             "Error while trying to play sound, sound name {} not recognized".format(
#                 sound_name,
#             )
#         )
#         return

#     # check to make sure file exists
#     if not path.exists(fname):
#         logging.error(
#             "Error while trying to play sound, sound {} not found in {}".format(
#                 sound_name, fname
#             )
#         )

#         # try one more time from parent directory
#         logging.warning("Checking for sound {} in {}".format(sound_name, fname))
#         fname = "../" + fname
#         if not path.exists(fname):
#             logging.error(
#                 "Error while trying to play sound, sound {} not found in {}".format(
#                     sound_name, fname
#                 )
#             )
#             return

#     # play async sound
#     # sound effects are os dependent
#     if platform.system() == "Windows":
#         winsound.PlaySound(fname, winsound.SND_ASYNC | winsound.SND_ALIAS)

