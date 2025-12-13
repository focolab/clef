import StimBaseClass

print('Not done! Needs to be refactored to match InvCoreLDIPolygon e.g. use pycromanager')

class TorstoscopeBLSPolygon(StimBaseClass):
    def __init__(self, args, local_handles={}):

        # call superconstructor
        super().__init__(args, local_handles)

        # scope params
        self.DSI_IMGWIDTH = 912
        self.DSI_IMGHEIGHT = 1140
        self.polygon_dims = [self.DSI_IMGWIDTH, self.DSI_IMGHEIGHT]

        # initialize connection to polygon and light source
        polygon_logfile = self.savedir + "_polygon_output.txt"
        bls_logfile = self.savedir + "_bls_output.txt"
        self.polygon_process, self.polygon_socket = initialize_polygon(
            polygon_logfile
        )
        self.bls_process, self.bls_socket = initialize_bls(bls_logfile)

        # get hardcoded calibration points for polygon
        (pcx, pcy, icx, icy) = self.get_Polygon_calibration_points()
        self.calibration_points = {"pcx": pcx, "pcy": pcy, "icx": icx, "icy": icy}
        self.pcx = np.array(pcx)
        self.pcy = np.array(pcy)
        self.icx = np.array(icx)
        self.icy = np.array(icy)

        
        # initialize a full-field mask
        self.full_field_stim_mask = np.ones(shape=self.polygon_dims)
        self.user_submitted_mask = np.zeros(shape=self.polygon_dims)
        self.spool()

        # send static mask to polygon
        send_mask_to_polygon(self.polygon_socket, self.user_submitted_mask)


        # skipping old logic of using static stim roi vs dynamic stim roi

    # function to precompile e.g. numba functions
    def spool(self):

        # spool jitted functions with dummy routine
        tstart = time.time()

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

        tend = time.time()
        logging.info(
            "Spooling wbliveStimClass functions took {}s".format(tend - tstart)
        )

    def close(self):
        try:
            logging.debug("Closing polygon and bls subprocesses")
            self.polygon_process.terminate()
            self.bls_process.terminate()
            self.polygon_socket.close()
            self.bls_socket.close()
        except Exception as err:
            logging.exception(
                "Error while trying to close down stimulus interface: {}".format(
                    err
                )
            )


    def get_metadata(self):

        # return base class metadata
        metadata = super().get_metadata()
        print('GOTTA TEST METADATA INHERITANCE')

        # add more attributes
        metadata["stim_intensity_list"] = self.stim_intensity_list
        metadata["calibration_points"] = self.calibration_points


    def get_Polygon_calibration_points(self):
        """ load calibraiton points for polygon """ 
        raise(Exception('Torstoscope BLS Polygon calibration not implemented yet!'))



# function to initialize polygon connection
def initialize_polygon(
    logfile_fname,
    PORT=5007,
    path_to_exe="C:/Users/confocal/source/repos/polygon-app/x64/Debug/polygon-app.exe",
):

    try:

        # open logfile and pass to subprocess
        f = open(logfile_fname, "w")

        if path_to_exe:
            polygon_process = subprocess.Popen(path_to_exe, stdout=f)
            # self.polygon_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, creationflags=0x08000000)

    except Exception as err:

        logging.exception("Error during launching of polygon app: {}".format(err))
        raise (err)

    # connect to polygon app
    try:

        HOST = "localhost"
        polygon_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        polygon_socket.connect((HOST, PORT))
        logging.debug("Connected to Polygon server app on port {}".format(PORT))

    except Exception as err:
        logging.debug("Error while trying to connect to polygon app: {}".format(err))

    return polygon_process, polygon_socket


# function to initialize connection to bioled light source controller
def initialize_bls(
    logfile_fname,
    PORT=5008,
    path_to_exe="C:/Users/confocal/source/repos/bls-app/x64/Debug/bls-app.exe",
):

    try:

        # open logfile and pass to subprocess
        f = open(logfile_fname, "w")

        if path_to_exe:
            bls_process = subprocess.Popen(path_to_exe, stdout=f)

    except Exception as err:
        logging.exception("Error during launching of bls app: {}".format(err))
        raise (err)

    try:

        HOST = "localhost"
        bls_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bls_socket.connect((HOST, PORT))
        logging.debug("Connected to BLS server app on port {}".format(PORT))

    except Exception as err:
        logging.exception("Error while trying to connect to bls app: {}".format(err))
        raise (err)

    return bls_process, bls_socket

def send_mask_to_polygon(s, mask, preprocessed=False):

    logging.debug("Sending mask of shape {} to polygon.".format(mask.shape))

    if not preprocessed:
        mask = mask.astype(np.bool).T
        mask = np.flip(mask, axis=0)
        mask = np.flip(mask, axis=1)
        mask = np.packbits(mask)

    # send message
    try:

        s.send(mask)

        # grab response and reshape
        # buff_size = 1024
        # resp = s.recv(buff_size)
        # returned = np.frombuffer(res, dtype=np.bool).reshape(msg.shape)

        # print
        # logging.debug("Response: {}".format(resp))

    except Exception as err:
        logging.exception(
            "Exception while trying to send/recv a message to/from the polygon: {}".format(
                err
            )
        )
        raise (err)


# function to send a control signal to bioled light source controller
# modulates channel and brightness in %0.1 units (so 1000 = 100%)
def send_bls_control_command(s, channel, value):

    # write message
    msg = [channel, value]
    msg = np.array(msg)
    msg = msg.tobytes()

    try:
        s.send(msg)

        # grab response and reshape
        # buff_size = 1024
        # resp = s.recv(buff_size)
        # returned = np.frombuffer(res, dtype=np.bool).reshape(msg.shape)

        # print
        # logging.debug("Response from bls: {}".format(resp))

    except Exception as err:
        logging.exception(
            "Exception while trying to send/recv a message to/from the BLS controller: {}".format(
                err
            )
        )
        raise (err)
