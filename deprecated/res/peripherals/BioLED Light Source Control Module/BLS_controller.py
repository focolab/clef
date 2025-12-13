import numpy as np
import napari
import tifffile as tf
import socket
import sys
import subprocess

# flags
DEBUG = False
launch_exe = 1
PORT = 5008

# to launch exe externally
if launch_exe:
    path_to_exe='C:/Users/confocal/source/repos/bls-app/x64/Debug/bls-app.exe'
    bls_process = subprocess.Popen(path_to_exe, stdout=subprocess.PIPE)

# connect to cpp program
if not DEBUG:
    HOST = "localhost"
    s_bls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_bls.connect((HOST, PORT))

def bls(channel, value, DEBUG=False):

    # write message
    msg = [channel, value]
    msg = np.array(msg)
    msg = msg.tobytes()
    s_bls.send(msg)

    # grab response and reshape
    buff_size = 1024
    #resp = s_bls.recv(buff_size)
    # returned = np.frombuffer(res, dtype=np.bool).reshape(msg.shape)

    # print
    #print("Response: {}".format(resp))


# initialize 
bls(1, 0)


