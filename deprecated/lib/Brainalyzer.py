import time, logging, random
import numpy as np

# for qtvisualizer
from multiprocessing import Pipe, shared_memory

# import custom modules
try:
    from lib import BrainalyzerWorker
except ImportError:
    import BrainalyzerWorker # for local testing

# ignore numpy warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

# sys.path.append('C:/Users/rldun/Desktop/brainalyzer/lib')

class Brainalyzer:
    def __init__(self, args, local_handles={}):

        # get general params
        self.args = args
        self.rec_id = self.args["id"]
        self.roi = self.args["roi"]
        self.frames_to_grab = self.args["gooey_args"]["total_frames"]
        self.microscope_name = self.args["gooey_args"]["microscope_name"]
        self.saveroot = self.args.get("saveroot", "") 
        self.zsize = self.args["gooey_args"]["zsize"]
        self.xsize = self.args["roi"][2]
        self.ysize = self.args["roi"][3]
        self.camera_binning = None
        self.dtype = self.args.get("dtype", np.uint16)
        self.local_handles = local_handles

        # data transmission with subprocess ipc. default to shm
        self.ipc = self.args.get("data_ipc", "shared_memory")

        # set behavior mode vs neural imaging mode
        self.GUI_mode = self.args["gooey_args"].get("GUI_mode", 'neural_imaging')

        # if we're in behavior mode, initialize behavior-related hardware e.g. stage
        if self.GUI_mode == "behavior":
            self.initialize_behavior_mode()

        # holders
        self.events = []
        self.current_event = None
        self.stimulus_is_on = False
        self.stim_param_list = []
        self.shared_frame_memory_list = []
        self.shared_ndarray_list = []

        # initialize the worker
        self.initialize_worker()


    def initialize_worker(self):

        # alg-specific params for subprocess
        # self.stim_diameter = int(self.args["gooey_args"]["stimulus_diameter"])
        self.stim_intensity_ops = self.args["gooey_args"]["stim_intensity_options"]
        self.stim_intensity = self.stim_intensity_ops[0]
        
        self.vis_args = {
            # "stim_diameter": self.stim_diameter,
            "id": self.rec_id,
            "saveroot": self.saveroot,
            "ysize": self.ysize,
            "xsize": self.xsize,

            "total_frames": self.frames_to_grab,
            "zsize": self.zsize,
            "stim_intensity": self.stim_intensity,
            "data_ipc": self.ipc,
            "GUI_mode": self.GUI_mode,
            "camera_binning": self.camera_binning,
        }

        # initialize the visualizer
        try:

            # create a pipe to visualizer
            self.parent_conn, self.child_conn = Pipe()

            if self.ipc == 'shared_memory':

                # make a shared memory buffer and associated ndarray for each z plane
                for z in range(self.zsize):
                    try: 
                        shared_frame_memory = shared_memory.SharedMemory(
                            create=True,
                            size=self.ysize * self.xsize * 2,
                            name="shared_frame_memory_{}".format(z),
                        )

                    # bad cleanup means file might already exist, in which case just keep using it
                    except FileExistsError:
                        shared_frame_memory = shared_memory.SharedMemory(name="shared_frame_memory_{}".format(z), create=False, size=self.ysize * self.xsize * 2)

                    shared_ndarray = np.ndarray(
                        shape=(self.ysize, self.xsize),
                        buffer=shared_frame_memory.buf,
                        dtype=self.dtype,
                    )

                    self.shared_frame_memory_list.append(shared_frame_memory)
                    self.shared_ndarray_list.append(shared_ndarray)

                # also make frame counter
                try: 
                    self.shared_image_count = shared_memory.ShareableList(
                        [0], name="shared_image_count"
                    )
                except FileExistsError:
                    # shared_memory.ShareableList(name='shared_image_count').shm.unlink()
                    # shared_memory.ShareableList(name='shared_image_count').shm.close()
                    self.shared_image_count = shared_memory.ShareableList(None, name="shared_image_count") # None is equivalent to create=false?


            else:
                raise(Exception('ERROR ipc must be shared_memory'))
            
            # create process and start it
            self.proc = BrainalyzerWorker.BrainalyzerWorker(self.child_conn, self.vis_args)
            self.proc.start()

        except Exception as err:
            logging.error('Error while initializing BrainalyzerWorker: {}'.format(err))
            raise(err)
        
    def initialize_behavior_mode(self):
        
        if self.args["gooey_args"]["microscope_name"] == "innovation core thunderscope":
            # shared memory for xy stage control
            self.shared_stage_offset_xy = shared_memory.ShareableList(
                [0, 0], name="shared_stage_offset_xy"
            )

            # data structure for xy stage control memory
            self.xy_stage_position_list = []

        # micromanager handle for stage control
        self.mmc = self.local_handles.get('mmc', None)
        if self.mmc is None:
            logging.warning('Error while initializing GUI_mode: behavior, no local handle to micromanager found')
        else:
            cam = self.mmc.getCameraDevice()
            self.camera_binning = self.mmc.getProperty(cam, "Binning")

    def get_xy_offset(self):

        # grab offset values
        offsetx, offsety = self.shared_stage_offset_xy

        # if either value is nonzero
        if offsetx or offsety:

            # reset offsets
            self.shared_stage_offset_xy[0] = 0
            self.shared_stage_offset_xy[1] = 0

            # return
            return int(offsetx), int(offsety)
        else:
            return None
        
    def sync_stage(self):

        # store stage position
        if self.GUI_mode == 'behavior' and self.microscope_name == "innovation core thunderscope" and self.args["id"] != "test":
            self.xy_stage_position_list.append([self.mmc.getXPosition(), self.mmc.getYPosition()])

            # get xy offset
            xy_offset = self.get_xy_offset()
            if xy_offset is not None:

                # maybe apply smoothening here
                # print('stage offset: ({}, {}))'.format(xy_offset[0], xy_offset[1]))

                # move stage
                self.mmc.setRelativeXYPosition(xy_offset[0], xy_offset[1])
            

    def initialize_model(self):
        """Initialize the model and return the model object"""

        # dummy function in this case... just seed rng for reproducibility
        fname_root = self.args["id"]
        random.seed(fname_root)

    def get_metadata(self, args=None):
        """Return metadata for this alg captured during runtime"""

        metadata={
            'stim_param_list': self.stim_param_list,
        }

        # optional metadata
        if self.GUI_mode == 'behavior' and self.microscope_name == "innovation core thunderscope":
            metadata['xy_stage_position_list'] = self.xy_stage_position_list

        return metadata
    
    
    def skip(self):
        """ utility fxn """
        pass

    def process_sample(self, img, zndx):
        """ what do you want this alg to do with each frame"""

        # store the frame in shared memory
        self.store_frame_in_shm(img, zndx)

        # update image count (frames not volumes)
        self.shared_image_count[0] += 1

        # also see if there are any updates from gui
        data = self.get_event()
        if data is not None:
            self.events.append(data)
            logging.info('Brainalyzer::process_volume> storing event: {}'.format(data))
            self.current_event = data

        # adjust stage if necessary
        if self.GUI_mode == 'behavior':
            self.sync_stage()

    def process_volume(self): 
        """ after a full z-scan, what do you want this alg to do with the volume """
        pass
    
    def check_stim(self, image_ndx, cooldown_counter=0):
        """ algorithm for receiving stimulus events from worker, and formulating them for stim_interface """

        # TODO HERE WE NEED TO PROCESS THIS OBJECT AND ADD TO IT IF NEED BE

        # initialize output var
        stim_params = {}
        new_cooldown = 0

        # grab current event if it's present
        if not self.stimulus_is_on and self.current_event is not None:

            # get stim params from randomly initialized lists or from event
            stim_intensity = self.current_event["stim_intensity"]

            # add data to stim params
            # trigger stim_on on next volume
            zndx = image_ndx % self.zsize

            # if we've done a full volume which happens on zndx=zsize-1
            stim_on = image_ndx + self.zsize - zndx

            # trigger event-based stimulation one volume later to allow for polygon to update
            # stim_on += self.zsize # consider taking this out

            # brainalyzer can emit pulsed or continuous stimulation
            stim_event = self.current_event
            event_type = stim_event['event_type']

            # build stim parameters
            stim_params["stim_on"] = stim_on
            stim_params["stim_intensity"] = stim_intensity
            stim_params["event"] = stim_event

            # handle different types of stim event signals
            if event_type == 'pulse-rect-roi-list' or event_type == 'full-field-button':

                # add delay if need be
                # stim_on += self.current_event["stim_delay_vols"] * self.zsize

                # set number of stim in frames
                num_stim_frames = self.current_event["stim_duration_vols"] * self.zsize
                stim_params["stim_off"] = stim_on + num_stim_frames

                # check if recording is about to end, in which case we don't want to stimulate
                if stim_params["stim_off"] > self.frames_to_grab:

                    # remove stimulus info
                    stim_params = {}

            elif event_type == 'stream-rect-roi-list' or 'stream-widefield':

                # a la hammer of dawn
                self.stimulus_is_on = True
                self.current_event = None

            else:
                logging.critical('check_stim> event type {} not recognized!!'.format(event_type))

            # reset current event
            self.current_event = None

            # set new cooldown timeout, in this case we're ignoring because it's taken care of by gui
            # new_cooldown = self.frames_to_cool_down_after_stim
            new_cooldown = 0
            self.stim_param_list.append(stim_params)

        # if we are still stimulating but no new event, update neuron positions
        elif self.stimulus_is_on and self.current_event is None:

            # todo for if our rois are moving during a stimulus stream
            # currently roi xys are sent in the event. if we want to change this, we need to instead use e.g. shmem as ipc to keep mutable roi state
            # note that for this we'd also need to change InvCoreLDIPolygon::process_multi_rect_stream_event
            pass

        # if we receive an event while stimulation is ongoing, it'll be to tell us to stop stimulating
        elif self.stimulus_is_on and self.current_event is not None:

            # add data to stim params
            # trigger stim_on on next volume
            zndx = image_ndx % self.zsize

            # if we've done a full volume which happens on zndx=zsize-1
            stim_off = image_ndx + self.zsize - zndx

            # set
            stim_params["stim_off"] = stim_off
            stim_params["event"] = self.current_event

            # set holder
            self.stimulus_is_on = False
            self.current_event = None

            # update most recent stim param with accompanying info
            # so will only contain on/off frames and start xy
            self.stim_param_list[-1]["stim_off"] = stim_off

        if stim_params: logging.info('Brainalyzer::check_stim> emitting stim params: {}'.format(stim_params))
        return stim_params, new_cooldown
    
    def plot_model(self, show_plot=False, savefilename=None):
        """Plot current state of model"""

        logging.warning("There is no alg model to plot!")

    def close(self):
        """Close the worker process"""

        # close the shared memory buffers
        for shm in self.shared_frame_memory_list:
            shm.close()
            shm.unlink()
        self.shared_image_count.shm.close()
        self.shared_image_count.shm.unlink()
        
        # close the worker process
        self.parent_conn.send("close")

        # possibly close out stage-related e.g. shared memory
        if self.GUI_mode == 'behavior':
            self.shared_stage_offset_xy.shm.close()
            self.shared_stage_offset_xy.shm.unlink()


    #########################################################################################
    # internal methods
    #########################################################################################

    def get_event(self):
        """ check worker to see if there's any events to process """

        # if anything in pipe, return it, otherwise implicitly returns None
        if self.parent_conn.poll():
            data = self.parent_conn.recv()
            return data

    def store_frame_in_shm(self, img, zndx):

        # get the shared memory buffer for this z plane
        self.shared_ndarray_list[zndx][:] = img[:]



# standalone testing
if __name__ == "__main__":

    import tifffile as tf

    # load tiff file for streaming
    # fname = "C:/Users/rldun/Desktop/temp_render/20230904-15-59-40/20230904-15-59-40.tiff"
    # fname = "C:/Users/rldun/Desktop/temp_render/test.tiff"
    # fname = "E:/RLD/20220116_RLD_1/20220116-19-12-47/20220116-19-12-47.tiff"
    # fname = 'E:/Data/Kato lab/RLD/20240204_RLD_1/20230921-17-15-33_test.tiff'
    # fname = 'C:/Users/rldun/Desktop/temp_render/20240628/_1/_1_MMStack_Pos0.ome.tif'
    # fname = 'D:/Kato Lab/RLD/20240628/_1/_1_MMStack_Pos0.ome.tif'

    # fname_root = '20240705-14-10-06'
    fname_root = '20230904-15-59-40'

    # datadir = 'C:/Users/rldun/data/TEMP_DATA_HOLDER/20240705_1/'
    # datadir = 'C:/Users/rldun/Desktop/temp_render/20230904-15-59-40/'
    # fname = datadir + fname_root + '/' + fname_root + '.tiff'
    # fname = datadir + fname_root + '.tiff'
    datadir = 'C:/Users/rldun/Desktop/temp_render/'
    fname = datadir + '20240628_wb_gcamp_test.tiff'

    # load data
    data = tf.imread(fname)
    # tiff = tf.TiffFile(fname)
    # data = tiff.pages[0].asarray()

    # initialize subprocess
    vps = 5
    # sample_zsize = 4 # numz we're simulating, can be less than original data numz
    data_zsize = 12  # original data numz 
    sample_zsize = 12
    # data_zsize = 12
    total_vols = 250
    cooldown_counter = 0
    
    fps = sample_zsize * vps
    vis_args = {
        "id": "test",
        "saveroot": "",
        "roi": [600, 1360, 2000, 480],
        # "roi": [0, 0, 2048, 2048],
        # "roi": [0, 0, 1024, 1024],
        # "stim_diameter": 30,
        "ysize": data.shape[1],
        "xsize": data.shape[2],
        # "ysize": data.shape[0],
        # "xsize": data.shape[1],
        "gooey_args":{
            "total_frames": total_vols * sample_zsize,
            "zsize": sample_zsize,
            # "stimulus_diameter": 30,
            "stim_intensity_options": [10],
            "microscope_name": "innovation core thunderscope",
            "GUI_mode": "neural_imaging",
            # "GUI_mode": "behavior",

        },
        "zsize": sample_zsize,
        "stim_duration_vols": 4,
        "stim_intensity": 10,
        # "camera_binning": "2x2"
    }
    alg = Brainalyzer(vis_args)

    # pre load list of frames (if streaming) - this is needed for bigger files
    # frame_list = [tiff.pages[i].asarray() for i in range(total_vols*zsize)]

    # initialize visualizer with blank frame
    # frame = data[0, :, :]
    # alg.update_display_images(frame)

    #  display images for some amount of time
    # zmip = np.zeros((data.shape[1], data.shape[2], zsize), dtype=np.uint16)
    # zmip = np.zeros((data.shape[1], data.shape[2]), dtype=np.uint16)

    # reshape data to support subsampling in z (20240421)
    if data.ndim == 3:
        # tyx -> tzyx
        data_tsize = data.shape[0] // data_zsize
        data = data.reshape([data_tsize, data_zsize, data.shape[1], data.shape[2]])

    elif data.ndim == 4:
        # tyxz
        pass
    else:
        raise(Exception('Error: data dimensionality {} not supported'.format(data.shape)))

    time.sleep(5)

    print('Beginning stream...')

    # iterate volumes
    for i in range(0, total_vols - 1):
    # for i in range(0, total_vols):

        # if linear indexing, take zmip
        for z in range(0, sample_zsize):

            # 20240421
            img = data[i, z, :, :]

            # zmip[:, :, z] = data[i * zsize + z, :, :]
            # zmip = np.maximum(zmip, data[i * zsize + z, :, :])
            image_ndx = i * sample_zsize + z
            # img = data[image_ndx, :, :]

            # img = data[i, :, :, z]
            # img = tiff.pages[i * zsize + z].asarray()
            # img = frame_list[i * zsize + z]

            # mip = zmip.max(axis=2)
            # vis.update_display_images(mip)
            alg.process_sample(img, zndx=z)

            # pause for fps sim
            if fps is not None:
                time.sleep(1 / fps)

        # get events from vis
        # event = alg.get_event()
        # if event is not None:
        # print(event)

        # every volume, have the alg process it
        alg.process_volume()

        # update triggering
        stim_params, cooldown_counter = alg.check_stim(
            image_ndx, cooldown_counter
        )
        if stim_params:
            print(stim_params)



    # close
    alg.close()
