import os, json, logging
from pyqtgraph.Qt import QtWidgets, QtCore
import os, time
import pyqtgraph as pg
import numpy as np
import skimage.io
from cv2 import resize
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter
from skimage.measure import centroid, label
from tierpsy.analysis.ske_create.getSkeletonsTables import getWormMask
from tierpsy.analysis.ske_create.segWormPython import mainSegworm
import scipy


try:
    from lib.models import BrainalyzerModel # production
except ModuleNotFoundError:
    try:
        from models import BrainalyzerModel # debugging
    except ModuleNotFoundError:
        import BrainalyzerModel # standalone debugging


class HeadCurvatureAndXYStageModel(BrainalyzerModel.BrainalyzerModel):

    def __init__(self, brainalyzer_worker, **kwargs):

        # store vars
        self.brainalyzer_worker = brainalyzer_worker
        self.xsize = self.brainalyzer_worker.xsize
        self.ysize = self.brainalyzer_worker.ysize
        self.frame_centroid = (0, 0)
        self.skel_points, self.prev_skeleton = None, None
        self.behavior_mode = 'multi-worm'
        self.behavior_mode_options = ['multi-worm', 'single-worm']
        self.image_to_render = None
        self.first_img_flag = True
        self.camera_binning = self.brainalyzer_worker.camera_binning

        # initialization for behavior mode multi work aka open-loop
        self.num_pulses = 3
        self.pulse_duration = 5
        self.inter_pulse_interval = 25
        self.pulse_f0 = None
        self.pulse_onsets = []
        self.pulse_offsets = []

        # intialization for behavior mode single worm aka closed-loop
        self.display_binary_mask = False
        self.swap_head_tail = False
        self.downsample_factor = 2
        self.cy, self.cx = self.ysize//2, self.xsize//2
        self.threshold_percentile = 1.
        self.lower_threshold = None
        self.circle_mask = None
        self.num_head_points = 5
        self.head_displacement = None
        self.head_displacement_plot_tsize = 200
        self.recent_head_displacements = np.full((self.head_displacement_plot_tsize), np.nan)

        # stage calibration at 5x -> FOV should say 4.78
        self.micron_to_pix_ratio = 100./74.
        if self.camera_binning == "1x1":
            self.camera_binning_multiplier = 1
        elif self.camera_binning == "2x2": 
            self.camera_binning_multiplier = 2
        else:
            self.camera_binning_multiplier = 1
        self.stage_dampening_factor = 0.5

        # save some model features for reconstruction after recording finished
        self.head_displacement_threshold_val_list = []
        self.head_displacement_list = []
        self.running_tstep = []

        # decorating
        self.color_dict = {
            "red": "#DC143C",
            "green": "#009900",
            "grey": "#A9A9A9",
            "orange": "#FFA500",
        }

        # load superconstructor
        super().__init__(brainalyzer_worker, **kwargs)

    def load_model(self):
        """ load model, context dependent (deserialize, load from file, connect to url etc) """
        pass

    # def get_current_image_count(self):
    # return self.brainalyzer_worker.image_count

    def get_model_dict(self):
        
        # try and get some info from brainalyzer_worker
        rec_id = self.brainalyzer_worker.rec_id
        saveroot = self.brainalyzer_worker.saveroot
        
        # nothing pre-saved in this model
        model = {
            'rec_id': rec_id,
            'saveroot': saveroot,
            'downsample_factor': self.downsample_factor,
            'num_head_points': self.num_head_points,
            'head_displacement_plot_tsize': self.head_displacement_plot_tsize,
            'micron_to_pix_ratio': self.micron_to_pix_ratio,
            'camera_binning': self.camera_binning, # '1x1', '2x2
            'camera_binning_multiplier': self.camera_binning_multiplier,
            'stage_dampening_factor': self.stage_dampening_factor,
            # 'num_pulses': self.num_pulses,
            # 'pulse_duration': self.pulse_duration,
            # 'inter_pulse_interval': self.inter_pulse_interval,
            'behavior_mode': self.behavior_mode,
            'display_binary_mask': self.display_binary_mask,
            'threshold_percentile': self.threshold_percentile,
            'head_displacement_threshold_val_list': self.head_displacement_threshold_val_list,
            'head_displacement_list': self.head_displacement_list,
            'running_tstep': self.running_tstep,
        }

        return model

    def save_model(self):
        """ save model, context dependent, info necessary to reconstruct model (serialize, save to file, connect to url etc) """
        
        # try and save model info for later debugging
        try:
            model_dict = self.get_model_dict()
            saveroot = model_dict.get('saveroot', '')
            model_output_fname = saveroot + '_HeadCurvatureAndXYStageModel_output.json'

            with open(model_output_fname, 'w') as f:
                json.dump(model_dict, f, indent=4, sort_keys=True)
            logging.info('Saved model output to: {}'.format(model_output_fname))
            
        except Exception as err:
            logging.warning('Failed to save model info! {}'.format(err))


    def predict_single_worm_mode(self):

        if self.first_img_flag:
            self.first_img_flag = False
            auto_levels = True
        else:
            auto_levels = False

        # grab image from brainalyzer_worker
        frame = self.brainalyzer_worker.img_list[0].T
        frame = np.flip(frame, axis=1)

        # downsample
        downsampled_frame = resize(frame, np.array(frame.shape) // self.downsample_factor)

        # get centroid
        filtered_frame = self.image_filter(downsampled_frame, lower_threshold=self.lower_threshold, circle_mask=self.circle_mask)
        self.frame_centroid = centroid(filtered_frame) * self.downsample_factor

        # get skeleton
        self.skel_points, self.prev_skeleton = self.get_centerline_and_skeleton(filtered_frame, prev_skeleton=self.prev_skeleton)
        if self.skel_points is not None:
            self.skel_points *= self.downsample_factor # why multiply by downsample factor? -> to get into RL coords
            if self.swap_head_tail:
                self.skel_points = self.skel_points[::-1]
            self.head_displacement = self.get_head_displacement(self.skel_points, self.num_head_points)
        else:
            self.head_displacement = None

        # update head displacement
        self.update_head_displacement_plot()

        # update stage correction
        self.update_stage_tracker_with_centroid(frame_centroid=self.frame_centroid)

        # update image
        if self.display_binary_mask:
            upsampled_filtered_image = resize(filtered_frame, frame.shape)
            self.ii.setImage(upsampled_filtered_image)
        else:
            self.ii.setImage(frame, autoLevels=auto_levels)

        # update centroid
        self.centroid_marker.setData(x=[self.frame_centroid[0]], y=[self.frame_centroid[1]])

        # update head and centerline
        if self.skel_points is not None:
            self.head_marker.setData(x=[self.skel_points[0,1]], y=[self.skel_points[0,0]])
            self.centerline_marker.setData(x=self.skel_points[1:,1], y=self.skel_points[1:,0])
        else:
            self.head_marker.setData(x=[], y=[])
            self.centerline_marker.setData(x=[], y=[])

        return self.check_stim()

    def predict_multi_worm_mode(self):

        # update image
        if self.first_img_flag:
            self.first_img_flag = False
            auto_levels = True
        else:
            auto_levels = False

        frame = self.brainalyzer_worker.img_list[0].T
        frame = np.flip(frame, axis=1)
        self.ii.setImage(frame, autoLevels=auto_levels)

        return self.check_stim()
    
    def check_stim(self):
        if self.behavior_mode == "multi-worm":
            return self.check_stim_pulse_train()
        elif self.behavior_mode == "single-worm":
            return self.check_stim_head_displacement()

    def check_stim_pulse_train(self): 

        # see if current frame is >= pulse onset, and < pulse offset, then pulse is active
        frame_count = self.get_current_frame_count()
        if self.pulse_f0 is not None:
            for i in range(len(self.pulse_onsets)):
                if frame_count >= self.pulse_onsets[i] and frame_count < self.pulse_offsets[i]:
                    return 1
        
        return 0
    
    def check_stim_head_displacement(self):

        # see if line is above or below 0
        threshval = self.head_displacement_thresh_line.value()
        if threshval > 0:
            if self.recent_head_displacements[-1] > threshval:
                return 1
            else:
                return 0
        else:
            if self.recent_head_displacements[-1] < threshval:
                return 1
            else:
                return 0
    

    def update_head_displacement_plot(self):

        # update head displacement plot
        self.recent_head_displacements[:-1] = self.recent_head_displacements[1:]
        self.recent_head_displacements[-1] = self.head_displacement if self.head_displacement is not None else np.nan
        self.head_displacement_dataitem.setData(self.recent_head_displacements, connect='finite')

        # save head displacement for reconstruction
        self.running_tstep.append(self.get_current_frame_count())
        self.head_displacement_list.append(self.head_displacement)
        self.head_displacement_threshold_val_list.append(self.head_displacement_thresh_line.value())


    def update_stage_tracker_with_centroid(self, frame_centroid):
        
        # compute dxy given centroid
        if self.enable_stage_tracking_button.isChecked():
            dy, dx = self.cy - int(frame_centroid[0]), self.cx - int(frame_centroid[1])

            # print(self.camera_binning_multiplier)
            correction_x = int(dx *  self.micron_to_pix_ratio * self.camera_binning_multiplier * self.stage_dampening_factor)
            correction_y = int(dy * self.micron_to_pix_ratio * self.camera_binning_multiplier * self.stage_dampening_factor)
            # print('({}, {})'.format(correction_x, correction_y))

            # store stage correction
            self.brainalyzer_worker.shared_stage_offset_xy[0] = correction_y
            self.brainalyzer_worker.shared_stage_offset_xy[1] = correction_x

    def predict(self):
        """ inference """ 
        
        # get centroid
        if self.behavior_mode == "single-worm":
            return self.predict_single_worm_mode()

        elif self.behavior_mode == "multi-worm": 
            return self.predict_multi_worm_mode()
        
        else:
            return 0

    def get_circle_mask(self, frame):
        circle_radius = min(frame.shape[0]//2, frame.shape[1]//2)
        image_center = (frame.shape[0]//2, frame.shape[1]//2)
        Y, X = np.ogrid[:frame.shape[1], :frame.shape[0]]
        dist_from_center = np.sqrt((X - image_center[0])**2 + (Y-image_center[1])**2)
        mask = dist_from_center <= circle_radius
        return mask

    def image_filter(self, frame, lower_threshold=None, circle_mask=None):
        if lower_threshold is None:
            lower_threshold = np.percentile(frame, self.threshold_percentile)
        thresholded = np.zeros(frame.shape, frame.dtype)
        # thresholded[frame < lower_threshold] = 32767
        thresholded[frame < lower_threshold] = 65535

        if circle_mask is None:
            circle_mask = self.get_circle_mask(frame)
        circle_masked = thresholded*circle_mask

        # get single connectivity mask
        labeled = label(circle_masked)
        labels, counts = np.unique(labeled, return_counts=True)
        labels, counts = labels[1:], counts[1:]
        biggest_label_i = np.argmax(counts)
        biggest_label = labels[biggest_label_i]
        connectivity_mask = labeled==biggest_label
        connectivity_masked = circle_masked*connectivity_mask
        return connectivity_masked

    def get_centerline_and_skeleton(self, filtered_frame, prev_skeleton=None):
        np.float = float
        np.int = int
        np.object = object
        np.bool = bool

        threshold = 1
        _, worm_cnt, _ = getWormMask(filtered_frame, threshold, is_light_background=False)
        resampling_N = 49
        skel_args = {'num_segments' : 24, 'head_angle_thresh' : 60}
        if prev_skeleton is not None:
            skeleton_output_data = mainSegworm.getSkeleton(worm_cnt, prev_skeleton=prev_skeleton, resampling_N=resampling_N, **skel_args)
        else:
            skeleton_output_data = mainSegworm.getSkeleton(worm_cnt, resampling_N=resampling_N, **skel_args)
        skeleton, ske_len, cnt_side1, cnt_side2, cnt_widths, cnt_area = skeleton_output_data
        if len(cnt_side1) > 0:
            skel_points = (cnt_side1 + cnt_side2) / 2
            return skel_points, skeleton
        else:
            return None, None

    #compute head swing displacement
    def get_head_displacement(self, skel_points, num_head_points):
        p0x, p0y = skel_points[0]
        # p4x, p4y = skel_points[num_head_points-8] # closer to nose 
        # p5x, p5y = skel_points[num_head_points-10] # closer to neck
        p4x, p4y = skel_points[8] # closer to nose 
        p5x, p5y = skel_points[11] # closer to neck
        #vec_ neckline= [p5x-p4x, p5y-p4y]
        #vec_neckline_perp= [-(p5y-p4y), p5x-p4x]
        #vec_p4_to_p0 = [p4x-p0x, p4y-p0y]
        #project vec_p4_to_p0 onto vec_neckline_perp
        dot = -(p5y-p4y) * (p4x-p0x) + (p5x-p4x) * (p4y-p0y)
        projection_length= dot / np.linalg.norm([-(p5y-p4y), p5x-p4x])

        if self.neckbase_markers is not None:
            self.neckbase_markers.setData(x=[p4y, p5y], y=[p4x, p5x], pen=pg.mkPen(color=self.color_dict['orange'], width=2), brush=pg.mkBrush(color=self.color_dict['orange']), size=6, symbol='o')

        return projection_length

    def create_model_panel(self):

        # initialize gui items
        self.initialize_gui()

        # add overhead label
        self.model_panel_graphics_layout.addLabel('Model Description: {}'.format(self.get_model_description()), row=0, col=0, colspan=6)
        self.model_panel_graphics_layout.addLabel('Model Instructions: {}'.format(self.get_model_inference_instructions()), row=1, col=0, colspan=6)

        # add viewbox for rendering image
        rowspan_offset = 2
        model_viewbox_rowspan = 12
        self.model_panel_graphics_layout.addItem(self.vb, row=rowspan_offset, col=0, colspan=model_viewbox_rowspan)
        rowspan_offset += model_viewbox_rowspan

        # combobox for singleworm vs multiworm
        self.model_panel_graphics_layout.addLabel('Behavior Mode:', row=rowspan_offset, col=0, rowspan=1)
        self.model_panel_graphics_layout.addLabel('(MW) num pulses:', row=rowspan_offset, col=1, rowspan=1)
        self.model_panel_graphics_layout.addLabel('(MW) pulse duration (vol):', row=rowspan_offset, col=2, rowspan=1)
        self.model_panel_graphics_layout.addLabel('(MW) inter-pulse interval (vol):', row=rowspan_offset, col=3, rowspan=1)

        # move to next row -> behavior mode combobox
        rowspan_offset += 1
        self.model_panel_graphics_layout.addItem(self.behavior_mode_combobox_proxy_widget, row=rowspan_offset, col=0, colspan=1)

        # gui elements for multi-worm mode
        self.model_panel_graphics_layout.addItem(self.num_pulses_spinbox_proxy_widget, row=rowspan_offset, col=1, colspan=1)
        self.model_panel_graphics_layout.addItem(self.pulse_duration_spinbox_proxy_widget, row=rowspan_offset, col=2, colspan=1)
        self.model_panel_graphics_layout.addItem(self.inter_pulse_interval_spinbox_proxy_widget, row=rowspan_offset, col=3, colspan=1)
        self.model_panel_graphics_layout.addItem(self.pulse_train_button_proxy_widget, row=rowspan_offset, col=4, colspan=1)

        # gui elements for single-worm mode
        rowspan_offset += 1
        self.model_panel_graphics_layout.addItem(self.display_binary_mask_button_proxy_widget, row=rowspan_offset, col=1, colspan=1)
        self.model_panel_graphics_layout.addItem(self.swap_head_tail_button_proxy_widget, row=rowspan_offset, col=2, colspan=1)
        self.model_panel_graphics_layout.addItem(self.threshold_percentile_input_proxy_widget, row=rowspan_offset, col=3, colspan=1)
        self.model_panel_graphics_layout.addItem(self.enable_stage_tracking_button_proxy_widget, row=rowspan_offset, col=4, colspan=1)

        # move to next row
        rowspan_offset += 1

        # add plotitem
        self.model_panel_graphics_layout.addItem(self.head_displacement_plotitem, row=rowspan_offset, rowspan=4, col=0, colspan=2)
        rowspan_offset += 4
        # gui element for percentile threshold
        # self.model_panel_graphics_layout.addLabel('single-worm percentile:', row=2, col=7, rowspan=1)
        
        # add spacers to get labels and objects the right scaling
        # r = 1
        # while r <= rowspan_offset + r:
        self.model_panel_graphics_layout.addLabel('', row=rowspan_offset, col=0, rowspan=1)

    def behavior_mode_combobox_callback(self):

        # set behavior mode reference
        self.behavior_mode = self.behavior_mode_options[self.behavior_mode_combobox.currentIndex()]

        # change what gui elements are displayed
        if self.behavior_mode == 'multi-worm':
            pass
        elif self.behavior_mode == 'single-worm':
            pass


    def get_current_frame_count(self):
        return self.brainalyzer_worker.image_count
    
    def enable_stage_tracking_button_callback(self):

        # set color
        if self.enable_stage_tracking_button.isChecked():
            self.enable_stage_tracking_button.setStyleSheet("background-color: {}".format(self.color_dict['orange']))
        else:
            self.enable_stage_tracking_button.setStyleSheet("background-color: {}".format(self.color_dict['grey']))


    def initialize_gui(self):

        # render image in panel
        # viewbox <- imageitem <- image
        self.vb = pg.ViewBox(lockAspect=True, enableMouse=True, border=None, enableMenu=True)
        # self.vb.invertY()
        # self.vb.invertX()
        # self.vb.setLimits(xMin=0, xMax=self.ysize, yMin=0, yMax=self.xsize)
        self.ii = pg.ImageItem()
        # self.ii.setImage(self.brainalyzer_worker.img_list[0]) # will this throw error if called with empty array? like if model is initialized before images have hit shm for brainalyzer_worker 
        self.vb.addItem(self.ii)

        # item for centroid
        self.centroid_marker = pg.ScatterPlotItem(x=[self.xsize//2], y=[self.ysize//2], pen=None, brush=(255, 0, 0), size=10, symbol='+')
        self.vb.addItem(self.centroid_marker)

        # item for head
        self.head_marker = pg.ScatterPlotItem(x=[self.xsize//2], y=[self.ysize//2], pen=None, brush=(0, 255, 0), size=10, symbol='o')
        self.vb.addItem(self.head_marker)

        # item for center line
        self.centerline_marker = pg.ScatterPlotItem(x=[self.xsize//2], y=[self.ysize//2], pen=None, brush=(255, 0, 0), size=5, symbol='o')
        self.vb.addItem(self.centerline_marker)

        # combobox for singleworm vs multiworm
        self.behavior_mode_combobox = pg.ComboBox(items=self.behavior_mode_options)
        self.behavior_mode_combobox.setEditable(True) # for center align 
        self.behavior_mode_combobox_lineEdit = self.behavior_mode_combobox.lineEdit()
        self.behavior_mode_combobox_lineEdit.setAlignment(QtCore.Qt.AlignCenter) # for center align
        self.behavior_mode_combobox_lineEdit.setReadOnly(True)
        self.behavior_mode_combobox.textActivated.connect(self.behavior_mode_combobox_callback)
        self.behavior_mode_combobox.setCurrentIndex(0) # -1 because 0-indexed

        # wrap in proxy widget
        self.behavior_mode_combobox_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.behavior_mode_combobox_proxy_widget.setWidget(self.behavior_mode_combobox)

        ## stuff for multi-worm mode
        self.num_pulses_spinbox = QtWidgets.QSpinBox()
        self.num_pulses_spinbox.setRange(1, 1000)
        self.num_pulses_spinbox.setValue(self.num_pulses)
        self.pulse_duration_spinbox = QtWidgets.QSpinBox()
        self.pulse_duration_spinbox.setRange(1, 1000)
        self.pulse_duration_spinbox.setValue(self.pulse_duration)
        self.inter_pulse_interval_spinbox = QtWidgets.QSpinBox()
        self.inter_pulse_interval_spinbox.setRange(1, 1000)
        self.inter_pulse_interval_spinbox.setValue(self.inter_pulse_interval)

        # connect to callbacks
        self.num_pulses_spinbox.valueChanged.connect(self.num_pulses_spinbox_callback)
        self.pulse_duration_spinbox.valueChanged.connect(self.pulse_duration_spinbox_callback)
        self.inter_pulse_interval_spinbox.valueChanged.connect(self.inter_pulse_interval_spinbox_callback)

        # create proxy widgets for each spinbox
        self.num_pulses_spinbox_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.num_pulses_spinbox_proxy_widget.setWidget(self.num_pulses_spinbox)
        self.pulse_duration_spinbox_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.pulse_duration_spinbox_proxy_widget.setWidget(self.pulse_duration_spinbox)
        self.inter_pulse_interval_spinbox_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.inter_pulse_interval_spinbox_proxy_widget.setWidget(self.inter_pulse_interval_spinbox)

        # button for activating pulse train
        self.pulse_train_button = QtWidgets.QPushButton('Start Pulse Train')
        self.pulse_train_button.clicked.connect(self.start_pulse_train_button_callback)
        self.pulse_train_button_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.pulse_train_button_proxy_widget.setWidget(self.pulse_train_button)

        ## stuff for single-worm mode

        # button for toggling binary mask display
        self.display_binary_mask_button = QtWidgets.QPushButton('Toggle Binary Mask Display')
        self.display_binary_mask_button.clicked.connect(self.display_binary_mask_button_callback)
        self.display_binary_mask_button_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.display_binary_mask_button_proxy_widget.setWidget(self.display_binary_mask_button)

        # button for swapping head and tail
        self.swap_head_tail_button = QtWidgets.QPushButton('Swap Head and Tail')
        self.swap_head_tail_button.clicked.connect(self.swap_head_tail_button_callback)
        self.swap_head_tail_button_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.swap_head_tail_button_proxy_widget.setWidget(self.swap_head_tail_button)

        # input for percentile
        self.threshold_percentile_input = QtWidgets.QLineEdit(str(self.threshold_percentile))
        self.threshold_percentile_input.returnPressed.connect(self.threshold_percentile_change_callback)
        self.threshold_percentile_input_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.threshold_percentile_input_proxy_widget.setWidget(self.threshold_percentile_input)

        # pushbutton for activating stage tracking
        self.enable_stage_tracking_button = QtWidgets.QPushButton("Enable stage tracking")
        self.enable_stage_tracking_button.setCheckable(True)
        self.enable_stage_tracking_button.toggled.connect(self.enable_stage_tracking_button_callback)
        self.enable_stage_tracking_button.setStyleSheet("background-color: {}".format(self.color_dict['grey']))
        self.enable_stage_tracking_button_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.enable_stage_tracking_button_proxy_widget.setWidget(self.enable_stage_tracking_button)

        # plot for head displacement
        self.head_displacement_plotitem = pg.PlotItem()
        self.head_displacement_plotitem.disableAutoRange()
        self.head_displacement_plotitem.setRange(xRange=[0, self.head_displacement_plot_tsize], yRange=[-400, 400])
        self.head_displacement_plotitem.setLabel(axis='bottom', units='recent frames')
        self.head_displacement_plotitem.setTitle('head displacement')
        self.head_displacement_dataitem = self.head_displacement_plotitem.plot([], pen=pg.mkPen(self.color_dict['green'], width=2))

        # lines for head displacement threshold
        self.head_displacement_thresh_line = self.head_displacement_plotitem.addLine(x=None, y=2, pen=pg.mkPen(self.color_dict['red'], style=QtCore.Qt.DashLine), movable=True)

        # neckbase marker 
        self.neckbase_markers = pg.ScatterPlotItem(x=[self.xsize//2], y=[self.ysize//2], pen=pg.mkPen(color='#CC5500', width=2), brush=pg.mkBrush(color='#CC5500'), size=6, symbol='o')
        self.vb.addItem(self.neckbase_markers)

    def start_pulse_train_button_callback(self):
        
        # print('starting pulse train')
        self.pulse_f0 = self.get_current_frame_count()

        # given starting frame, create list of frames which should be pulsed
        # print('frame count: {}'.format(self.pulse_f0))
        self.pulse_onsets = [self.pulse_f0 + i*(self.pulse_duration + self.inter_pulse_interval) for i in range(self.num_pulses)]

        # make list of frames when pulse is active
        self.pulse_offsets = [self.pulse_f0 + i*(self.pulse_duration + self.inter_pulse_interval) + self.pulse_duration for i in range(self.num_pulses)]

    def num_pulses_spinbox_callback(self):
        self.num_pulses = self.num_pulses_spinbox.value()

    def pulse_duration_spinbox_callback(self):
        self.pulse_duration = self.pulse_duration_spinbox.value()

    def inter_pulse_interval_spinbox_callback(self):
        self.inter_pulse_interval = self.inter_pulse_interval_spinbox.value()

    def display_binary_mask_button_callback(self):
        self.display_binary_mask = not self.display_binary_mask

    def swap_head_tail_button_callback(self):
        self.swap_head_tail = not self.swap_head_tail

    def threshold_percentile_change_callback(self):
        self.threshold_percentile = float(self.threshold_percentile_input.text())
        self.lower_threshold = None

    def close(self):
        self.save_model()

## for debugging
def advance_state(self):

    # for whatever state count (e.g. frame), set debugger holder to hold the correct frame
    self.img_list[0] = self.frame_array[self.state_count,:,:]
    self.image_count = self.state_count


def add_debugging_statevars(bug, frame_array):

    # store tyx frame array
    bug.frame_array = frame_array
    bug.ysize = frame_array.shape[1]
    bug.xsize = frame_array.shape[2]
    bug.image_count = 0
    bug.camera_binning = "2x2"

    # store first frame of video in data structure which model has access to
    bug.img_list = [frame_array[0,:,:]]
    bug.shared_stage_offset_xy = [0, 0]

    # set incremenet state function
    bug.set_advance_state_function(advance_state)

# standalone testing
if __name__ == '__main__':
    
    import tifffile as tf

    # load example data
    # fname = '/home/jackbo/data/20240628_RLD_1/_1/_1_MMStack_Pos0.ome.tif'
    # fname = 'D:/Kato Lab/RLD/20240628/_1/_1_MMStack_Pos0.ome.tif'
    fname_root = '20240705-14-10-06'

    datadir = 'C:/Users/rldun/data/TEMP_DATA_HOLDER/20240705_1/'
    fname = datadir + fname_root + '/' + fname_root + '.tiff'
    
    if not os.path.exists(fname):
        print('No file found at {} did you mean to change the test data file path?'.format(fname))
    frame_array = tf.imread(fname)

    # set up args for model and debugger
    model_args = {}
    debugger_args = {
        'timestep_seconds': 0.1,
        'rec_id': 'test',
        'saveroot': 'C:/Users/rldun/code/wb-live/analysis/notebooks/20230519 analysis/modeling/data',
    }
    num_t = 200


    ##############################################################################################

    # # imports
    import BrainalyzerModelDebugger

    # # instantiate debugger object
    bug = BrainalyzerModelDebugger.BrainalyzerModelDebugger(debugger_args=debugger_args)

    # define advance_state function (+ necessary internal vars) and, add to debugger
    add_debugging_statevars(bug, frame_array)

    # instantiate model object with debugger instead of brainalyzer_worker as object
    json_fname = 'head_curvature_and_xystage.json'
    mod = HeadCurvatureAndXYStageModel.from_json(json_fname, bug, **model_args)

    # link model panel to debugger
    bug.link_model_panel(mod)

    # iterate states
    print('running HeadCurvatureAndXYStageModel')
    for i in range(0, num_t):

        bug.advance_state()
        mod.predict()

        # iterate frames "states"
        if i % 100 == 0:
            print(i)

    bug.close()
    mod.close()
