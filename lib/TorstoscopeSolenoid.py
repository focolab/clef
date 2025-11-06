import StimBaseClass

print('Not done! Needs to be refactored to e.g. use pycromanager')

class TorstoscopeSolenoid(StimBaseClass):

    def __init__(self, args, local_handles={}):

        # superconstructor
        super().__init__(args, local_handles)

        # load handle to microscope hardware
        self.mmc = self.get_mmc()

        # intialize solenoid with constant ontime
        # in milliseconds
        offtime = 500000  # this number doesn't matter because we're not using the arduino to sequence our stims
        ontime = 500000  # this number is effectively "constantly on"
        cmd = "F,{},{}".format(offtime, ontime)
        send_serial_command(cmd, self.mmc)


        
    @abstractmethod
    def submit_stim_params(self, stim_params, image_ndx):
        pass


    @abstractmethod
    def process_stim_params_event(self, stim_params, image_ndx):
        pass


    def activate_stim(self):
        
        # send serial command (always starts with high voltage on solenoid)
        cmd = "E"
        send_serial_command(cmd, self.mmc)


    def inactivate_stim(self):

        # send serial command to turn off (state when there is no voltage to the solenoid))
        cmd = "D"
        send_serial_command(cmd, self.mmc)



def send_serial_command(command, mmc, port="COM11"):

    # useful debug output
    logging.debug("Sending serial command: {} to port: {}.".format(command, port))

    endln = "\r"
    mmc.setSerialPortCommand(port, command, endln)
