if __name__ == "__main__":

    from gooey import Gooey, GooeyParser
    import time
    import ClosedLoopEngine
    import os
    import sys
    import json
    import logging

    print('cwd: {}'.format(os.getcwd()))

    logging.basicConfig(level=logging.DEBUG)

    # identify if a previous config exists in this directory
    fnames = os.listdir()
    if "gooey_config_reload.json" in fnames:
        load_build_config = "gooey_config_reload.json"
        logging.debug(">>> Loading local gooey_config.json!")
    else:
        load_build_config = None

    

    @Gooey(
        program_name="CLEF launcher",
        default_size=(900, 800),
        tabbed_groups=True,
        dump_build_config=True,
        load_build_config=load_build_config,
        # image_dir="C:/Users/Leica/code/wb-live/wb-live-v4/res/img",
        image_dir = "./res/img"
    )
    def main():

        # initialize overall gooey object
        desc = "Application for launching closed-loop experiment!"
        parser = GooeyParser(description=desc)

        # add argument elements that get added to the gui
        acq_group = parser.add_argument_group(
            "Acquisition controls", "Settings for experimental acquisition."
        )
        acq_group.add_argument(
            "output_folder",
            help="Destination folder for saving data",
            widget="DirChooser",
        )
        acq_group.add_argument(
            "-d",
            "--total-frames",
            default=30000,
            type=int,
            help="Duration (in frames) for the acquisition",
        )
        acq_group.add_argument(
            "-cfg",
            "--mm-configuration-file",
            choices=[
                "C:\\Program Files\\Micro-Manager-2.0beta\\200212_PRIME_BSI_TIE.cfg",
                "C:\\Program Files\\Micro-Manager-2.0beta\\MMConfig_demo.cfg",
                "C:\\Program Files\\Micro-Manager-2.0beta\\20201016_PRIME_BSI_TIE.cfg",
                "C:\\MMConfigs\\MMConfig_demo.cfg",
                "C:\\MMConfigs\\CSUW1.cfg",
                "C:\\MMConfigs\\CSUW1-LDI-Polygon.cfg"
            ],
            default="C:\\MMConfigs\\CSUW1-LDI-Polygon.cfg",
            help="Micro-Manager configuration file",
        )
        acq_group.add_argument(
            "-z", "--zsize", default=10, type=int, help="Number of z-planes for imaging"
        )
        acq_group.add_argument(
            "-mip", "--save-mip", action="store_true", help="Save MIP video"
        )
        acq_group.add_argument(
            "-strobe",
            "--strobe-acquisition",
            action="store_true",
            help="Strobe illumination during acquisition?",
        )
        acq_group.add_argument(
            "-sifi",
            "--strobe-inter-frame-interval",
            type=int,
            default=80,
            help="Strobed inter-frame interval in ms (must be > exposure!)",
        )
        acq_group.add_argument(
            "-struct",
            "--save-structural-scan",
            choices=[
                "none",
                "GFP + RFP (torstoscope)",
                "GFP + RFP (torstoscope) 1um steps",
                "NeuroPAL (innovation core) 1um steps",
            ],
            default="none",
            help="Save structural scan?",
        )

        # group for experimental info
        expr_group = parser.add_argument_group(
            "Expr. metadata", "Metadata for experiment."
        )
        expr_group.add_argument(
            "-strn", "--subject-strain", help="Subject strain", required=True
        )
        expr_group.add_argument(
            "-preprx",
            "--subject-condition",
            choices=[
                "",
                "ATR",
                "LTM 1x buffer T0",
                "LTM 1x butanone T0",
                "LTM 3x buffer T0",
                "LTM 3x buffer T16",
                "LTM 3x butanone T0",
                "LTM 3x butanone T16",
            ],
            default="",
            help="Subject expr. condition",
        )
        expr_group.add_argument(
            "-atr",
            "--atr-concentration",
            type=float,
            help="Concentration of ATR (uM)",
            required=True,
        )
        expr_group.add_argument(
            "-zstp",
            "--z-step-size",
            type=float,
            help="Step size in Z dimension",
            required=True,
        )
        expr_group.add_argument(
            "-nose",
            "--nose-orientation",
            choices=["left", "right", "other"],
            help="Nose orientation",
            required=True,
        )
        expr_group.add_argument(
            "-vnc",
            "--vnc-orientation",
            choices=["up", "down", "other"],
            help="VNC orientation",
            required=True,
        )
        expr_group.add_argument("-egg", "--num-eggs", type=int, help="Number of eggs", default=0)
        expr_group.add_argument(
            "-app",
            "--microscope-name",
            choices=["torstoscope spinning disk", "innovation core spinning disk", "innovation core thunderscope"],
            default="innovation core spinning disk",
            help="Microscope apparatus",
        )
        expr_group.add_argument(
            "-notes", "--experimental-notes", help="experimental notes", required=True
        )

        # group for closed-loop parameters
        closed_loop_group = parser.add_argument_group(
            "Closed-loop controls",
            "Customize the closed-loop experimental acquisition options",
        )
        closed_loop_group.add_argument(
            "-alg",
            "--trigger-algorithm",
            choices=[
                "Dynamic range deriv",
                "RoiDeriv",
                "StimOnsetFromList",
                "Dummy algorithm (does nothing)",
                "PointAndClick",
                "HammerOfDawn",
                "Brainalyzer",
            ],
            default="Brainalyzer",
            help="Closed-loop algorithm to trigger stimulation",
        )
        closed_loop_group.add_argument(
            "-gui",
            "--GUI-mode",
            choices=[
                'neural_imaging',
                'behavior'
            ],
            default='neural_imaging',
            help='GUI mode'
        )
        closed_loop_group.add_argument(
            "-bl",
            "--rec-baseline",
            default=0,
            type=int,
            help="Frames at beginning of recording where stimulations are not permitted",
        )
        closed_loop_group.add_argument(
            "-smp",
            "--save-alg-model-plot",
            action="store_true",
            help="Save CL algorithm data plot",
        )

        # params for stimulus settings
        stim_settings_group = parser.add_argument_group(
            "Stimulus settings",
            "Controls for stimulus e.g. LMM5-controlled lasers or arduino-controlled microfluidic valves",
        )
        stim_settings_group.add_argument(
            "-stint",
            "--stim-interface",
            choices=[
                "LMM5_561",
                "Polygon1000_470",
                "Polygon1000_590",
                "InvCore-LDI-Polygon-640",
                "solenoid",
                "InvCore-ThunderscopeLED3",
                "InvCore-SpinningDisk-639",
                "no stim",
            ],
            default="InvCore-LDI-Polygon-640",
            help="Stimulus interface",
        )
        stim_settings_group.add_argument(
            "-stroi",
            "--use-static-stim-roi",
            action="store_true",
            help="Upload initial static ROI (for Polygon1000, requires struct. images)",
        )
        stim_settings_group.add_argument(
            "-sfo",
            "--frames-to-stimulate-for-options",
            default="48",
            help="Frames to stimulate for, can be list with multiple comma-separated entries",
        )
        stim_settings_group.add_argument(
            "-sio",
            "--stim-intensity-options",
            default="10",
            help="Stimulation intensity (0-100%), can be list with multiple comma-separated entries",
        )
        stim_settings_group.add_argument(
            "-sdia",
            "--stimulus-diameter",
            widget="IntegerField",
            default=10,
            help="If using PointAndClick, stim ROI diameter",
        )

        # specific params for each closed-loop algorithm
        # cl_alg_group = parser.add_argument_group(
        #     "Closed-loop algorithm params",
        #     "Settings specific to closed-loop stimulation algorithms",
        # )
        # cl_alg_group.add_argument(
        #     "-stp",
        #     "--stim-threshold-pos",
        #     type=float,
        #     default=0.06,
        #     help="Positive threshold to trigger stimulus",
        # )
        # cl_alg_group.add_argument(
        #     "-stn",
        #     "--stim-threshold-neg",
        #     type=float,
        #     default=0.06,
        #     help="Negative threshold to trigger stimulus (abs)",
        # )
        # cl_alg_group.add_argument(
        #     "-scd",
        #     "--stim-cooldown",
        #     default=900,
        #     type=int,
        #     help="Refractory period (in frames) following stimulation before next stim",
        # )
        # cl_alg_group.add_argument(
        #     "-ssp",
        #     "--skip-stimulation-probability",
        #     default=0.1,
        #     type=float,
        #     help="Probability of skipping stimulation",
        # )
        # cl_alg_group.add_argument(
        #     "-sdp",
        #     "--delay-stimulation-probability",
        #     default=0.4,
        #     type=float,
        #     help="Probability of delaying stimulation",
        # )
        # cl_alg_group.add_argument(
        #     "-sdo",
        #     "--stim-delay-frames-options",
        #     default="200, 400",
        #     help="If stimulation is delayed, frames to delay for. Can be list with multiple comma-separated entries",
        # )
        # cl_alg_group.add_argument(
        #     "-userstimlist",
        #     "--stim-onset-list-options",
        #     help="Frames for fixed stimulus onset (for StimOnsetFromList), should be comma-separated frames",
        # )

        # group for development options
        dev_group = parser.add_argument_group(
            "Dev ops",
            "Useful options for ongoing development, testing, and evaluation of wb-live",
        )
        dev_group.add_argument(
            "-i",
            "--input-recording",
            help="File to simulate recording",
            widget="FileChooser",
        )
        dev_group.add_argument(
            "-be",
            "--acquisition-backend",
            choices=[
                'micromanager compiled MMCorePy bindings',
                'micromanager pymmcore',
                'pycromanager'
            ],
            default='pycromanager',
            help='Acquisition backend'
        )
        dev_group.add_argument(
            "-nsi",
            "--no-save-images",
            action="store_true",
            help="DO NOT save output image data",
        )
        dev_group.add_argument(
            "-nsm",
            "--no-save-metadata",
            action="store_true",
            help="DO NOT save output metadata",
        )
        dev_group.add_argument(
            "-sgd",
            "--save-gooey-defaults",
            action="store_true",
            help="Save gooey config settings",
        )
        dev_group.add_argument(
            "-pfwb",
            "--prefill-wb-ops",
            action="store_true",
            help="Prefill wboptions.mat and meta.mat",
        )
        dev_group.add_argument(
            "-sms",
            "--send-sms",
            action="store_true",
            help="Send SMS to Ray when done",
        )

        # get "command line" arguments from GUI
        args = parser.parse_args()

        # display all arguments
        gooey_args = vars(args)
        logging.debug("Incoming gooey arguments: {}".format(gooey_args))
        time.sleep(1)

        # write gooey args to config dump for next
        if gooey_args["save_gooey_defaults"]:
            save_gooey_arg_build(gooey_args)

        # format args if necessary, this line might be redundant because of mutable data structures
        gooey_args = format_gooey_args(gooey_args)

        # send all args to acq engine
        ClosedLoopEngine.launch_wblive_from_gooey(gooey_args)

    def format_gooey_args(gooey_args):
        """Turn gooey args that offer lists of multiple values into proper lists"""

        try:
            args_keys = gooey_args.keys()
            for k in args_keys:
                if k.endswith("options"):

                    # if there's no options supplied (empty prompt), append empty list
                    if gooey_args[k] is None:
                        gooey_args[k] = []

                    # otherwise should be comma-delimited string
                    else:
                        argslist = gooey_args[k].split(",")
                        gooey_args[k] = [int(a) for a in argslist]

            return gooey_args

        except Exception as err:
            print("Error while formatting gooey data: {}".format(err))
            print("Please make sure text input is formatted correctly!")
            raise (err)

    def save_gooey_arg_build(gooey_args, fname="gooey_config.json"):

        # open gooey_config
        try:
            logging.info("Saving gooey defaults to {}.".format(fname))
            res = json.load(open(fname, "r"))
        except FileNotFoundError:
            print(
                "No gooey config named {} found! Not writing last save.".format(fname)
            )
            return

        # iterate settings and manipulate mutable data elements
        for tab in res["widgets"]["gooey-setup.py"]["contents"]:
            for item in tab["items"]:
                arg_name = item["data"]["display_name"]
                curr_default = item["data"]["default"]
                new_default = gooey_args[arg_name]

                # switch default to new value
                logging.debug(
                    "Changing {} default from {} to {}".format(
                        arg_name, curr_default, new_default
                    )
                )
                item["data"]["default"] = gooey_args[arg_name]

        # save
        try:
            with open(fname, "w") as f:
                f.write(json.dumps(res, indent=2))
        except Exception as err:
            print(
                "Error while trying to write new default arguments to {}: {}.".format(
                    fname, err
                )
            )

    # if __name__ == "__main__":

    main()
