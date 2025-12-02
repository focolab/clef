import sys, time, json, logging, os.path
import numpy as np
import tifffile as tf
from datetime import datetime

# custom libraries and utils
from lib import MMSubroutines
# from lib import wbliveStimClass
from lib import StimBaseClass
from lib import wbliveUtils as utils

# trigger algorithms
# from lib import DynamicRangeDeriv
# from lib import RoiDeriv
from lib import DummyAlg
# from lib import StimOnsetFromList
# from lib import PointAndClick
# from lib import HammerOfDawn
from lib import Brainalyzer


logging.basicConfig(level=logging.WARNING)

def launch_wblive_from_gooey(ops=None):

    if ops:

        args = {"gooey_args": ops}

    # if we're doing a dry run, return
    try:
        if ops:
            if ops["input_recording"] is not None:
                fname = ops["input_recording"]
                logging.debug("Simulating recording from file: {}".format(fname))

                # send all settings to closed-loop engine
                run_acquisition(args)
                return

    except Exception as err:
        logging.exception(
            "Problem while trying load data file to stimulate recording: {}".format(err)
        )
        raise (err)

    # if using pycro-manager, we'll send triggers to existing micromanager
    if args["gooey_args"]["acquisition_backend"] == "pycromanager":

        print("Running acquisition with pycromanager")
        try:

            # if we got settings from the frontend, add them to acq args
            run_acquisition(args)
        except Exception as err:
            print("Error while running acquisition: {}".format(err))
        finally:
            print("Done session")
            sys.exit()


def run_acquisition(args):

    # initialize objects for closed-loop
    # timing and local vars
    frame_time_list = []
    cooldown_counter = 0
    # vols_to_grab = frames_to_grab // zsize

    # get gooey params from args object
    try:
        args_string = json.dumps(args, indent=4, sort_keys=True)
        logging.debug(
            "Beginning acquisition with the following arguments: {}".format(args_string)
        )

        # todo: switch to args.get fxn for default values
        # tricky because what's a "default"...
        zsize = args["gooey_args"]["zsize"]
        frames_to_grab = args["gooey_args"]["total_frames"]
        # ENABLE_STIM = args["gooey_args"]["enable_closed_loop"]
        frames_baseline_window = args["gooey_args"]["rec_baseline"]
        savedir = args["gooey_args"]["output_folder"]
        NO_SAVE_IMAGES = args["gooey_args"]["no_save_images"]
        NO_SAVE_METADATA = args["gooey_args"]["no_save_metadata"]
        save_mip_movie = args["gooey_args"]["save_mip"]
        save_alg_model_plot = args["gooey_args"]["save_alg_model_plot"]
        STROBE_ACQUISITION = args["gooey_args"]["strobe_acquisition"]
        strobe_inter_frame_interval = args["gooey_args"]["strobe_inter_frame_interval"]
        config_file = args["gooey_args"]["mm_configuration_file"]
        IS_DEMO_ACQUISITION = config_file.endswith("MMConfig_demo.cfg")
        save_structural_scan = args["gooey_args"]["save_structural_scan"]
        PREFILL_WB_OPS = args["gooey_args"]["prefill_wb_ops"]
        NOTIFY_SMS_ON_DONE = args["gooey_args"].get("send_sms", True)
        trigger_alg = args["gooey_args"]["trigger_algorithm"]
        acquisition_backend = args["gooey_args"]["acquisition_backend"]
        GUI_mode = args["gooey_args"].get("GUI_mode", 'neural_imaging')

    except KeyError as err:
        logging.exception(
            "KeyError while initializing acquisition arguments: {}".format(err)
        )
        raise (err)

    # initialize output dir based on datetime
    dt = datetime.today().strftime("%Y%m%d-%H-%M-%S")
    savedir = savedir + "\\" + dt
    os.mkdir(savedir)
    saveroot = savedir + "\\" + dt
    args["id"] = dt
    args["saveroot"] = saveroot

    # intialize microscope object
    mmc = MMSubroutines.initialize_mmc(args, config_file)

    # fill gaps if/if not pycromanager
    if acquisition_backend == "pycromanager":
        res = mmc.getROI()
        roi = [res.getX(), res.getY(), res.getWidth(), res.getHeight()]
        args["roi"] = roi

    # set roi
    xsize = args["roi"][2]
    ysize = args["roi"][3]

    try:

        # CL alg selection goes here
        if trigger_alg == "Dynamic range deriv":
            alg = DynamicRangeDeriv.DynamicRangeDeriv(args)
        elif trigger_alg == "RoiDeriv":
            alg = RoiDeriv.RoiDeriv(args)
        elif trigger_alg == "StimOnsetFromList":
            alg = StimOnsetFromList.StimOnsetFromList(args)
        elif trigger_alg == "PointAndClick":
            alg = PointAndClick.PointAndClick(args)
        elif trigger_alg == "HammerOfDawn":
            alg = HammerOfDawn.HammerOfDawn(args)
        elif trigger_alg == 'Brainalyzer':
            alg = Brainalyzer.Brainalyzer(args, local_handles={"mmc": mmc})
        else:
            logging.debug("Running closed-loop with DUMMY algorithm")
            alg = DummyAlg.DummyAlg()

    except KeyError as err:

        logging.exception(
            "KeyError while initializing closed-loop algorithm: {}".format(err)
        )
        raise (err)

    except Exception as err:
        logging.exception(
            "Unknown exception during initialization of closed-loop algorithm: {}, exception type: {}".format(
                err, err.__class__.__name__
            )
        )

    # initialize alg before beginning experiment
    alg.initialize_model()
    
    # if IS_DEMO_ACQUISITION:
    #    MMSubroutines.prepare_demo_acquisition(mmc, args)
    # else:
    MMSubroutines.prepare_live_acquisition(mmc, args)
    # MMSubroutines.pre_flight_check(mmc, zsize)

    # do structural scans
    MMSubroutines.run_structural_scan(
        save_structural_scan, mmc, args, saveroot, dt, zsize
    )

    # intialize stimulus interface
    try:
        # stim = wbliveStimClass.wbliveStimClass(args, local_handles={"mmc": mmc})
        stim = StimBaseClass.StimBaseClass.initialize_stim_interface(args, local_handles={"mmc": mmc})
    except Exception as err:
        logging.exception(
            "Unknown {} exception during intialization of stimulus apparatus: {}".format(
                err.__class__.__name__, err
            )
        )

    try:

        # set some local vars for recording
        img_count = 0
        # xsize = args["roi"][2]
        # ysize = args["roi"][3]
        logging.info(
            "Beginning acquisition of [{},{},{}] frames.".format(
                frames_to_grab, ysize, xsize
            )
        )

        frames = np.zeros((frames_to_grab, ysize, xsize), dtype=np.uint16)

        t0 = time.time()

        FALSE_GRAB_COUNT = 0

        # if we're strobing, take our initial timestamp
        if STROBE_ACQUISITION:
            next_call = time.time()

            mmc.snapImage()
        else:
            
            frame_grab_t0 = time.time()
            mmc.stopSequenceAcquisition() # in case mm live is still running
            mmc.clearCircularBuffer()
            mmc.startContinuousSequenceAcquisition(0)

        # continuously grab frames until we're done
        while 1:

            # grab frames from camera buffer if they're available
            # big question: does snapimage cause mmc.getRemainingImageCount to incrememnt? or are those separate buffers
            rem = mmc.getRemainingImageCount()
            while rem > 0 or STROBE_ACQUISITION:

                # grab image
                try:
                    if STROBE_ACQUISITION:
                        img = mmc.getImage().astype(np.uint16)
                    else:
                        img = mmc.popNextImage().astype(np.uint16)
                        next_call = frame_grab_t0 # time we grabbed last frame

                # sometimes the circular buffer is empty? if so, retry while loop
                except Exception as err:
                    FALSE_GRAB_COUNT += 1
                    continue

                # image dimensions are different in pycromanager
                if acquisition_backend == "pycromanager":
                    img = img.reshape((ysize, xsize))

                # grab timestamp, add it out output structure, increment count
                frame_grab_t0 = time.time()
                frame_time_list.append(np.round(frame_grab_t0 - t0, decimals=4))
                frames[img_count, :, :] = img
                image_ndx = img_count
                img_count += 1

                if img_count % 200 == 0:
                    logging.info(
                        "Current image count: {}/{}".format(img_count, frames_to_grab)
                    )

                # if we're in cooldown, decriment cooldown counter
                if cooldown_counter > 0:
                    cooldown_counter -= 1

                # process frame
                zndx = image_ndx % zsize
                alg.process_frame(img, zndx)

                # update triggering, alg should return stim params
                stim_params, cooldown_counter = alg.check_stim(
                    image_ndx, cooldown_counter
                )

                # if we have stimulus parameters, submit to interface
                stim.submit_stim_params(stim_params, image_ndx)

                # # if we've done a full volume which happens on zndx=zsize-1
                if zndx == zsize - 1:

                    # filters need a few samples to work
                    # if image_ndx >= frames_baseline_window:

                    # process volume
                    # here we make a call that
                    # alg.process_volume()

                    # # update triggering, alg should return stim params
                    # stim_params, cooldown_counter = alg.check_stim(
                    #     image_ndx, cooldown_counter
                    # )

                    # # if we have stimulus parameters, submit to interface
                    # stim.submit_stim_params(stim_params, image_ndx)

                    # if we're in baseline period, just add zero for that volume to live data
                    # else:
                    #     alg.skip()

                    # function to progress stim state,
                    # e.g. turning on or turning off stim (indexed by subsuquent image)
                    # checking for next image
                    stim.check_stim(img_count)

                # exit if we've taken enough images
                if img_count == frames_to_grab:
                    raise (Exception("Done imaging!"))

                # if strobing, wait for next snap
                if STROBE_ACQUISITION:
                    nowtime = time.time()
                    next_call = next_call + strobe_inter_frame_interval / 1000

                    if next_call - nowtime < 0:
                        logging.warning(
                            "Strobe delay exceeded inter-frame-interval! Frame: {}".format(
                                image_ndx
                            )
                        )

                    else:
                        time.sleep(next_call - nowtime)

                    # start next image
                    mmc.snapImage()
                else:
                    rem = mmc.getRemainingImageCount()

    except Exception as err:

        logging.exception("Exception during acquisition: {}".format(err))

    finally:
        t1 = time.time()

        logging.info("Halting acquisition, which took {} seconds.".format(t1 - t0))

        # close down the appropriate things
        mmc.stopSequenceAcquisition()
        alg.close()
        stim.close()

        # take another structural scan if necessary
        if "NeuroPAL" in save_structural_scan:
            try:
                MMSubroutines.run_structural_scan(
                    save_structural_scan, mmc, args, saveroot, dt, zsize
                )
            except Exception as err:
                print(
                    "Error while taking structural scan at the end of the recording: {}".format(
                        err
                    )
                )

        # close down micro manager settings e.g. stage
        MMSubroutines.close(mmc, args)

        # send notification
        if NOTIFY_SMS_ON_DONE:
            interface = "twilio-sms"
            msg = "Your wb-live recording {} has completed.".format(args["id"])
            utils.notify(msg, interface=interface)

        if not NO_SAVE_IMAGES:
            MMSubroutines.saveScanTiffs(fname=saveroot + ".tiff", img_array=frames)

        if save_alg_model_plot:
            alg.plot_model(savefilename=saveroot + "_live_stim_fig.svg")

        if save_mip_movie:

            # set a fixed speedup according to exposure, default to 30ms
            if "exposure" in args:
                e = args["exposure"]
            else:
                e = 30

            utils.generate_mip_movie(
                savefilename=saveroot + "_mip_movie",
                frames=frames,
                zsize=zsize,
                exposure=e,
                GUI_mode=GUI_mode,
            )

        # grab and save metadata
        if not NO_SAVE_METADATA:
            metadata = args
            metadata["frame_time_list"] = frame_time_list
            metadata["t0"] = t0
            metadata["xsize"] = xsize
            metadata["ysize"] = ysize

            # alg and stim metadata
            # metadata is argument in case metadata entries require knownledge
            # of outside context
            metadata["alg_metadata"] = alg.get_metadata(args=metadata)
            metadata["stim_metadata"] = stim.get_metadata(args=metadata)

            # grab various settings
            metadata["mmc_metadata"] = MMSubroutines.get_metadata(
                args=metadata, mmc=mmc
            )

            # save
            utils.save_metadata(
                savefilename=saveroot + "_metadata.json", metadata=metadata
            )

            if PREFILL_WB_OPS:
                utils.prefill_wb_ops(savefileroot=savedir, metadata=metadata)

        # close down anything related to stimulus
        stim.close()
