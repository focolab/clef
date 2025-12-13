import serial


# class for connecting and issuing commands to KDS Gemini 88 Plus
class Gemini88P:
    def __init__(self, port="COM8", timeout=60, baudrate=9600):

        self.ser = self.connect(port=port, timeout=timeout, baudrate=baudrate)

    # function to open serial port connection
    def connect(self, port="COM8", timeout=60, baudrate=9600):

        # some hardcoded params, following provided code snippet
        parity = serial.PARITY_NONE
        stopbits = serial.STOPBITS_ONE
        bytesize = serial.EIGHTBITS

        # open port
        try:
            ser = serial.Serial(
                port,
                baudrate=baudrate,
                timeout=timeout,
                stopbits=stopbits,
                parity=parity,
                bytesize=bytesize,
            )
        except Exception as err:
            print("ERROR while trying to open serial port: {}".format(err))

        return ser

    # function to convert user command to ascii formatted command and send it to the device and display result
    def send_command(
        self,
        command_string,
        syringe="ab",
        command_params={"rate": 10, "units": "ul/min"},
    ):

        # get formatted command
        cmd = self._get_command(
            command_string,
            syringe=syringe,
            command_params={"rate": 10, "units": "ul/min"},
        )
        resp = self._send_command(cmd)

    # send command internal function
    def _send_command(self, cmd, print_response=True, read_length=200):

        ## append control carriers and encode
        to_send = cmd + "\r\n"
        msg = to_send.encode("ASCII")

        try:
            self.ser.write(msg)
        except Exception as err:
            print("ERROR while sending serial command: {}".format(err))

        # read return message
        try:
            res = self.ser.read(read_length)

            if print_response:
                print("req: {}\nresp: {}".format(msg, res.decode("ASCII")))
            return res

        except Exception as err:
            print("ERROR while receiving serial message: {}".format(err))

    # helper lookup table function for returning the appropriately formatted command
    def _get_command(self, command_string, syringe, command_params):

        # there's two pumps, A and B. To issue command to both, use AB
        if syringe != "a" and syringe != "b" and syringe != "ab":
            raise (Exception("ERROR syringe not recognized: {}".format(syringe)))

        # key/value pairs for desired command/formatted command
        try:
            cmd_lookup_table = {
                "get_version": "version",
                "get_syringe": "syr {}".format(syringe),
                "get_volume": "svolume {}".format(syringe),
                "stop": "stop {}".format(syringe),
                "get_rate": "crate {}".format(syringe),
                "get_rate_limits": "irate {} lim {}".format(
                    syringe, command_params["units"]
                ),
                "set_rate": "rate {} {} {}".format(
                    syringe, command_params["rate"], command_params["units"]
                ),
            }
        except Exception as err:
            raise (
                Exception(
                    "ERROR required command params not found for command string: {}, params: {}, error: {}".format(
                        command_string, command_params, err
                    )
                )
            )

        # if no command provided, list possible commands
        if command_string is None:
            print("Issuable commands: {}".format(cmd_lookup_table.keys()))
        else:

            # get command
            try:
                cmd = cmd_lookup_table[command_string]
                return cmd

            except Exception as err:
                raise (
                    Exception(
                        "ERROR command string not recognized: {}".format(command_string)
                    )
                )

    # get a medley of current settings for the pump
    def status(self):

        commands = [
            "get_version",
            "get_syringe",
            "get_volume",
            "get_rate",
            "get_rate_limits",
        ]

        # send a bunch of different commands and print response
        for c in commands:
            self.send_command(c)

    # function to display possible commands
    def get_command_options(self):
        self._get_command(command_string=None, syringe="ab", command_params={})

    # helper function to stop all syringe pumping
    def stop(self):
        self.send_command("stop", syringe="ab")

    # close our connection to the device
    def close(self):
        self.stop()
        self.ser.close()


#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
# example usage
# connect to hardware, you need to do this first time
pump = Gemini88P()

# display various info about device
pump.status()

# get list of commands we can issue to device
pump.get_command_options()

# send command
syringe = "a"
command = "set_rate"
command_params = {"rate": 10, "units": "ul/min"}
pump.send_command(command, syringe, command_params)

# stop all pumps
pump.stop()

# close when done
pump.close()