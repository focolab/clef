# for alg
import numpy as np
# import seaborn as sns
import logging, logging, warnings, time, os
from datetime import datetime

# for qtvisualizer
# from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
from multiprocessing import Process, shared_memory

# custom imports
try:
    from algorithms.models import BrainalyzerModel
    BRAINALYZER_DEBUG = False
except ModuleNotFoundError:
    from models import BrainalyzerModel # for local testing
    BRAINALYZER_DEBUG = True
    

# ignore numpy warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# pyqtgraph speedup
# pg.setConfigOption('useNumba', True)

class BrainalyzerWorker(Process):

    def __init__(self, child_conn, vis_args):

        
        super(BrainalyzerWorker, self).__init__()

        # stylizing
        self.vis_args = vis_args
        self.child_conn = child_conn

        if BRAINALYZER_DEBUG:
            self.model_dir = self.vis_args.get("model_dir", 'models/') # for local testing
            self.css_fname = self.vis_args.get("css_fname", "css/Ubuntu.qss") # for local testing
        else:
            self.model_dir = self.vis_args.get('model_dir', 'lib/models/')
            self.css_fname = self.vis_args.get("css_fname", "lib/css/Ubuntu.qss")

        # general params
        self.initialization_t0 = time.time()
        self.first_img_flag = True
        self.image_count = 0
        self.stim_refractory_vols = 0
        self.stim_duration_vols = self.vis_args.get("stim_duration_vols", 4)
        self.stimulus_is_on = False
        self.image_downsample = 1
        self.refractory_constant = 1
        self.rec_id = self.vis_args.get("id", "test")
        self.saveroot = self.vis_args.get("saveroot", "")
        self.ipc = self.vis_args.get("data_ipc", "shared_memory")
        self.dtype = self.vis_args.get("dtype", np.uint16)
        self.ysize = self.vis_args.get("ysize", None)
        self.xsize = self.vis_args.get("xsize", None)
        self.zsize = self.vis_args.get("zsize")
        self.total_frames = self.vis_args.get("total_frames", 48)
        self.total_vols = self.vis_args.get("total_frames") / self.zsize
        self.stim_diameter = self.vis_args.get("stim_diameter", 30)
        self.stim_intensity = self.vis_args.get("stim_intensity", 10)
        self.roi_plot_cmap = self.vis_args.get('roi_plot_cmap', "bright")
        self.GUI_mode = self.vis_args.get('GUI_mode', 'neural_imaging')
        self.camera_binning = self.vis_args.get("camera_binning", "1x1")
        # self.quant_cmap_list = [(255*np.array(x)).round().astype(np.uint8) for x in sns.color_palette(self.roi_plot_cmap)]
        self.quant_cmap_list = []  # BREAKING HOTFIX, PER IF SEABORN IS VALID DEPENDENCY
        self.num_rois_added = 0
        self.stim_cmap_list = [np.array((255, 0, 0), dtype=np.uint8)]
        self.shared_frame_memory_list = []
        self.img_list = []
        self.quant_roi_dict_list = []
        self.stim_roi_dict_list= []
        self.model_list = []
        self.vbox_z_label_list = []
        self.selected_model = None
        self.stop_rendering_image = False

        # convenience holder for coloring and stylizing params
        self.color_dict = {
            "red": "#DC143C",
            "green": "#009900",
            "grey": "#A9A9A9",
            "orange": "#FFA500",
        }


    def initialize_display(self):
        
        # initialize shared memory objects
        self.initialize_shm()

        # app <- layout <- vbox <- image
        # initialize qt app
        self.app = pg.Qt.mkQApp(name='BrainalyzerWorker')
        with open(self.css_fname, 'r') as f:
            self.app.setStyleSheet(f.read())

        # initialize image layout objects
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.image_vbox_list = []
        self.ii_list = []
        for z in range(self.zsize):

            # add label above image
            lab = pg.LabelItem(text='z={}'.format(z))
            self.vbox_z_label_list.append(lab)
            self.graphics_layout_widget.addItem(lab, row=0, col=z)

            # link axes together
            if z == 0:
                vb = pg.ViewBox(lockAspect=True, enableMouse=True, name='first_viewbox', border=None, enableMenu=True)
            else:
                vb = pg.ViewBox(lockAspect=True, enableMouse=True, border=None, enableMenu=True)
                vb.linkView(vb.XAxis, 'first_viewbox')
                vb.linkView(vb.YAxis, 'first_viewbox')

            # set viewbox can only pan within image
            vb.setLimits(xMin=0, xMax=self.ysize, yMin=0, yMax=self.xsize)
            vb.setBorder({'color': self.color_dict['grey'], 'width': 2})

            # create imageitem
            # ii = pg.ImageItem()
            ii = BrainalyzerImageItem(imageitem_id=z)
            ii.setImage(self.img_list[z]) # setting autolevels true on first pass is necessary for image to show

            # assemble objects/widgets and keep track of with holder lists
            vb.addItem(ii)
            self.graphics_layout_widget.addItem(vb, row=1, col=z)
            self.image_vbox_list.append(vb)
            self.ii_list.append(ii)

        # add graphics layout click callback
        self.graphics_layout_widget.scene().sigMouseClicked.connect(self.graphics_layout_mouse_clicked_callback)

        # histogram lut lookup
        self.lut_histo = pg.HistogramLUTItem(image=self.ii_list[0], fillHistogram=False, orientation='vertical', levelMode='mono')

        # signal proxy because we want to update plots in real time but ratelimit events
        self.histo_sig_proxy = pg.SignalProxy(self.lut_histo.sigLevelsChanged, rateLimit=3, slot=self.update_image_LUTs)
        self.lut_histo.setLevels(90, 150)
        self.lut_histo.setHistogramRange(60, 200) # this disables autorange
        self.graphics_layout_widget.addItem(self.lut_histo, row=1, col=self.zsize)


        ###########################################################
        # plots
        # add plotitem for roi data
        self.roi_plot_item = pg.PlotItem()
        self.roi_plot_item.disableAutoRange()
        self.roi_plot_item.setRange(xRange=[0, self.total_vols // 2], yRange=[90, 180])
        self.roi_plot_item.setLabel(axis='left', units='avg grey count')
        self.roi_plot_item.setLabel(axis='bottom', units='frame (vols)')
        self.roi_plot_item.setTitle('roi intensity')
         # 20240128 https://github.com/pyqtgraph/pyqtgraph/issues/2541 applies in current build

         # plotitem for roi derivs
        # self.roi_deriv_plot_item = pg.PlotItem()
        # self.roi_deriv_plot_item.disableAutoRange()
        # self.roi_deriv_plot_item.setRange(xRange=[0, self.total_vols // 2], yRange=[-10, 10])
        # self.roi_deriv_plot_item.setLabel(axis='left', units='g(t) - g(t-1)')
        # self.roi_deriv_plot_item.setLabel(axis='bottom', units='frame (vols)')
        # self.roi_deriv_plot_item.setTitle('roi intensity deriv')

        # add to layout
        colspan = (self.zsize+1) // 2
        if self.GUI_mode == 'neural_imaging':
            self.graphics_layout_widget.addItem(self.roi_plot_item, row=2, col=0, rowspan=1, colspan=colspan)
        # self.graphics_layout_widget.addItem(self.roi_deriv_plot_item, row=2, col=3, rowspan=1, colspan=self.zsize//4)

        ###########################################################
        # commmand panel
       # set command panel layout
        self.command_panel_layout = QtWidgets.QGridLayout()

        # button for intensity
        self.enter_stimulus_intensity_button = QtWidgets.QPushButton(
            "stim intensity (%): {}".format(self.stim_intensity)
        )
        self.enter_stimulus_intensity_button.clicked.connect(
            self.my_get_stimulus_intensity
        )
        
        # add button for adding/removing quant roi
        self.add_quant_roi_button = QtWidgets.QPushButton("add quant roi to z-plane 0")
        self.add_quant_roi_button.clicked.connect(self.add_quant_roi_to_image)
        self.delete_quant_roi_button = QtWidgets.QPushButton("delete quant roi from z-plane 0")
        self.delete_quant_roi_button.clicked.connect(self.delete_quant_roi_from_image)

        # button for adding/removing stimulus roi
        self.add_stim_roi_button = QtWidgets.QPushButton("add stim roi to z-plane 0")
        self.add_stim_roi_button.clicked.connect(self.add_stim_roi_to_image)
        self.delete_stim_roi_button = QtWidgets.QPushButton("remove stim roi from z-plane 0")
        self.delete_stim_roi_button.clicked.connect(self.delete_stim_roi_from_image)

        # add quant roi combobox
        self.quant_roi_combobox = pg.ComboBox(items=[str(x) for x in range(self.zsize)])
        self.quant_roi_combobox.setEditable(True) # for center align 
        self.quant_roi_combobox_lineEdit = self.quant_roi_combobox.lineEdit()
        self.quant_roi_combobox_lineEdit.setAlignment(QtCore.Qt.AlignCenter) # for center align
        self.quant_roi_combobox_lineEdit.setReadOnly(True)
        self.quant_roi_combobox.textActivated.connect(self.update_add_delete_quant_roi_button_text)

        # combobox for selecting z-plane to add stim roi to
        self.stim_roi_combobox = pg.ComboBox(items=[str(x) for x in range(self.zsize)])
        self.stim_roi_combobox.setEditable(True) # for center align
        self.stim_roi_combobox_lineEdit = self.stim_roi_combobox.lineEdit()
        self.stim_roi_combobox_lineEdit.setAlignment(QtCore.Qt.AlignCenter) # for center align
        self.stim_roi_combobox_lineEdit.setReadOnly(True)
        self.stim_roi_combobox.textActivated.connect(self.update_add_delete_stim_roi_button_text)
        
        # labels
        self.quant_roi_label = QtWidgets.QLabel("Add/Remove Quant ROIs")
        self.stim_label = QtWidgets.QLabel("Adjust Stimulus Params")
        self.stim_roi_label = QtWidgets.QLabel("Add/Remove Stim ROIs")
        self.pulse_stimulus_label = QtWidgets.QLabel("Pulse Stimulus")
        self.model_panel_label = QtWidgets.QLabel('Model Panel')

        # pulse buttons
        self.pulse_stimulus_rois_button = QtWidgets.QPushButton("pulse stimulate ROI(s)")
        self.pulse_stimulus_rois_button.clicked.connect(self.pulse_stimulus_rois)
        self.pulse_stimulus_rois_button.setStyleSheet("background-color: {}".format(self.color_dict['green']))
        self.pulse_full_field_button = QtWidgets.QPushButton("pulse stimulate full-field")
        self.pulse_full_field_button.clicked.connect(self.pulse_full_field)
        self.pulse_full_field_button.setStyleSheet("background-color: {}".format(self.color_dict['green']))
        
        # button for changing stim duration
        self.enter_stim_duration_vols_button = QtWidgets.QPushButton(
            "stimulus duration (vols): {}".format(self.stim_duration_vols)
        )
        self.enter_stim_duration_vols_button.clicked.connect(
            self.my_get_stimulus_duration_in_vols
        )

        ###########################################################
        # model stuff!

        # create layout and add it. model can then update that layout with plots and widgets (via proxy)
        row, col, rowspan, colspan = self.get_model_panel_localization_params()
        self.model_panel_graphics_layout = self.graphics_layout_widget.addLayout(row=row, col=col, rowspan=rowspan, colspan=colspan)

        # combobox for selecting model
        self.model_combobox = pg.ComboBox(items=[str(x.get_model_name()) for x in range(len(self.model_list))])
        self.model_combobox.setEditable(True) # for center align
        self.model_combobox_lineEdit = self.model_combobox.lineEdit()
        self.model_combobox_lineEdit.setAlignment(QtCore.Qt.AlignCenter) # for center align
        self.model_combobox_lineEdit.setReadOnly(True)
        self.model_combobox.textActivated.connect(self.select_model_from_combobox)

        # activate model state button
        self.activate_model_button = QtWidgets.QPushButton('activate model')
        self.activate_model_button.setCheckable(True)
        self.activate_model_button.setStyleSheet("background-color: {}".format(self.color_dict['green']))
        self.activate_model_button.toggled.connect(self.activate_model_button_callback)

        # model current output plot
        self.model_output_icon = QtWidgets.QLabel('Model Output: OFF')
        self.model_output_icon.setAlignment(QtCore.Qt.AlignCenter)
        self.model_output_icon.setStyleSheet("background-color: {}".format(self.color_dict['grey']))

        # arrangement of command panel
        self.command_panel_layout.addWidget(self.stim_label, 0, 0, alignment=QtCore.Qt.AlignCenter)
        self.command_panel_layout.addWidget(self.enter_stimulus_intensity_button, 1, 0)
        self.command_panel_layout.addWidget(self.enter_stim_duration_vols_button, 2, 0)
        self.command_panel_layout.addWidget(self.pulse_stimulus_label, 3, 0, alignment=QtCore.Qt.AlignCenter)
        self.command_panel_layout.addWidget(self.pulse_stimulus_rois_button, 4, 0)
        self.command_panel_layout.addWidget(self.pulse_full_field_button, 5, 0)

        self.command_panel_layout.addWidget(self.quant_roi_label, 0, 1, alignment=QtCore.Qt.AlignCenter)
        self.command_panel_layout.addWidget(self.quant_roi_combobox, 1, 1)
        self.command_panel_layout.addWidget(self.add_quant_roi_button, 2, 1)
        self.command_panel_layout.addWidget(self.delete_quant_roi_button, 3, 1)

        self.command_panel_layout.addWidget(self.stim_roi_label, 0, 2, alignment=QtCore.Qt.AlignCenter)
        self.command_panel_layout.addWidget(self.stim_roi_combobox, 1, 2)
        self.command_panel_layout.addWidget(self.add_stim_roi_button, 2, 2)
        self.command_panel_layout.addWidget(self.delete_stim_roi_button, 3, 2)

        self.command_panel_layout.addWidget(self.model_panel_label, 0, 3, alignment=QtCore.Qt.AlignCenter)
        self.command_panel_layout.addWidget(self.model_combobox, 1, 3)
        self.command_panel_layout.addWidget(self.activate_model_button, 2, 3)
        self.command_panel_layout.addWidget(self.model_output_icon, 3, 3) 

        ###########################################################

        # add text for frame number
        self.text_layout = QtWidgets.QGridLayout()
        self.vol_count_text = QtWidgets.QLabel(
            "frames: {}/{}".format(0, self.total_vols)
        )
        self.text_layout.addWidget(self.vol_count_text, 0, 0)

        # add text for stim refractory period
        self.stim_refractory_text = QtWidgets.QLabel(
            'stim refractory period: {}'.format(self.stim_refractory_vols)
        )
        self.stim_refractory_text.setAlignment(QtCore.Qt.AlignRight)
        self.text_layout.addWidget(self.stim_refractory_text, 0, 1)

        ###########################################################

        # containerization
        # assemble various layouts
        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.addWidget(self.graphics_layout_widget, 0, 0)
        # self.grid_layout.addWidget(self.model_textbox, 1, 1)
        self.grid_layout.addItem(self.command_panel_layout, 1, 0)
        self.grid_layout.addItem(self.text_layout, 2, 0)

        # add layout to window
        self.window = QtWidgets.QWidget()
        self.window.resize(1400, 800)
        self.window.setLayout(self.grid_layout)
        self.window.show()

    def initialize_brainalyzer_models(self):

        # load models, json files in model directory
        self.model_combobox.addItem('none', index=0)
        model_index_counter = 1
        for model_fname in os.listdir(self.model_dir):
            if model_fname.endswith('.json'):
                full_fname = os.path.join(self.model_dir, model_fname)
                try:
                    mod = BrainalyzerModel.BrainalyzerModel.from_json(full_fname, self)
                    self.model_list.append(mod)
                    self.model_combobox.addItem(mod.get_model_name(), index=model_index_counter)
                    model_index_counter += 1
                except Exception as err:
                    logging.error('Error while loading model {}: {}'.format(full_fname, err))
                    continue

    def get_model_panel_localization_params(self):
        rowspan = 1
        if self.GUI_mode == 'neural_imaging':
            col = (self.zsize+1) // 2
            colspan = int(np.ceil((self.zsize+1) / 2))
            row = 2
        elif self.GUI_mode == 'behavior':
            col = 2
            colspan = self.zsize + 1
            row = 1

        return row, col, rowspan, colspan
                    

    def initialize_shm(self):
        if self.ipc == 'shared_memory':
            # list of shared memory objects
            for z in range(self.zsize):
                self.shared_frame_memory_list.append(shared_memory.SharedMemory(name="shared_frame_memory_{}".format(z)))
                self.img_list.append(np.ndarray((self.ysize, self.xsize), dtype=self.dtype, buffer=self.shared_frame_memory_list[z].buf))
            
            # shared image count (frames not volumes)
            self.shared_image_count = shared_memory.ShareableList(name='shared_image_count')

            # xy stage position
            if self.GUI_mode == 'behavior':
                self.shared_stage_offset_xy = shared_memory.ShareableList(name="shared_stage_offset_xy")
        
        else:
            raise(Exception("Only shared_memory is supported for now"))
        return
    
    def configure_dashboard_before_run(self):
        if self.GUI_mode == "behavior":
            self.model_combobox.setText("head_curvature_and_xystage")
            self.select_model_from_combobox(True)

    def run(self):

        try:
            # initialize display, has to happen outside of init because
            self.initialize_display()
            self.initialize_brainalyzer_models()
            self.configure_dashboard_before_run()
            self.app.processEvents()
            self.image_count = self.shared_image_count[0]

            # main loop
            try:
                while self.image_count < self.total_frames:
                    try:
                        while self.image_count == self.shared_image_count[0]:
                            self.app.processEvents()
                    except ValueError as err:
                        continue

                    # update display
                    self.update_dashboard()

                    # sync image counts, can throw error unless we're stubborn :)
                    try:
                        self.image_count = self.shared_image_count[0]
                    except ValueError as err:
                        self.image_count = self.shared_image_count[0]
            
                    # events!
                    self.app.processEvents()
                
            except Exception as err:
                logging.error('Error while running BrainalyzerWorker: {}'.format(err))
                raise(err)
        except EOFError as err:
            logging.warning(
                "Visualization multiprocess socket has been closed EOF {}".format(err)
            )
        except BrokenPipeError as err:
            logging.warning(
                "Visualization multiprocess socket has been closed BrokenPipe {}".format(err)
            )
        except Exception as err:
            logging.error(
                "Unknown error in initialization of BrainAnalyzerWorker: {}".format(err)
            )
        finally:
            self.close()

    def close(self):

        # close model
        if self.selected_model is not None:
            self.selected_model.close()
        
        # close IPC
        if self.ipc == "shared_memory":
            self.shared_image_count.shm.close()
            for z in range(self.zsize):
                self.shared_frame_memory_list[z].close()

        self.child_conn.close()

    #####################################################################################
    # callbacks
    def graphics_layout_mouse_clicked_callback(self, event):
        
        # get nearby items
        items = self.graphics_layout_widget.scene().items(event.scenePos())
        
        # if anything is instance of imageitem, print its id
        selected_z = None
        for item in items:
            if isinstance(item, BrainalyzerImageItem):

                # set comboboxes according to clicked on image
                selected_z = item.imageitem_id
                self.stim_roi_combobox.setValue(str(selected_z))
                self.quant_roi_combobox.setValue(str(selected_z))
                self.update_add_delete_quant_roi_button_text(None)
                self.update_add_delete_stim_roi_button_text(None)
                break

        # update border color of viewboxes
        if selected_z is not None:
            self.update_viewbox_border_color(selected_z)
        # self.app.processEvents() # necessary for immediate update


    def my_get_stimulus_intensity(self, event):

        # grab value from user
        self.stimulus_intensity_dialog = QtWidgets.QInputDialog()

        # paramaterize dialog
        self.stimulus_intensity_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)

        self.stimulus_intensity_dialog.setIntMinimum(0)
        self.stimulus_intensity_dialog.setIntMaximum(100)

        # set event handler
        self.stimulus_intensity_dialog.intValueSelected.connect(
            self.set_stimulus_intensity
        )

        # show dialog
        self.stimulus_intensity_dialog.show()

    def set_stimulus_intensity(self, intensity):

        # set internal var
        self.stim_intensity = intensity

        # set button text
        self.enter_stimulus_intensity_button.setText(
            "stim intensity (%): {}".format(intensity)
        )

    def update_dashboard(self):

        # update images, plots, text
        self.update_display_images()
        self.update_display_text()
        self.update_model_prediction()
        self.update_refractory_counter()

        if self.GUI_mode == 'neural_imaging':
            self.update_quant_roi_plot()


    def update_refractory_counter(self):

        # refractory counter is in volumes
        if self.image_count % self.zsize == 0:

            # decrement refractory counter if need be
            if self.stim_refractory_vols >= 1:
                self.stim_refractory_vols -= 1

            if self.stim_refractory_vols == 1:
                self.check_stim_refractory_vols()

    def update_display_text(self):

        # update volume count
        if self.image_count % self.zsize == 0:
            self.vol_count_text.setText(
                "received vols: {}/{}".format(self.image_count//self.zsize, self.total_vols)
            )

            # update refractory volumes
            self.stim_refractory_text.setText(
                "stim refractory vols: {}".format(self.stim_refractory_vols)
            )

    def update_display_images(self):

        # get which frame to adjust
        z_ndx = self.image_count % self.zsize

        # update image
        if self.first_img_flag:
            auto_levels = True
            self.first_img_flag = False
        else:
            auto_levels = False
    
        # update image - at uint16
        if not self.stop_rendering_image:
            self.ii_list[z_ndx].setImage(self.img_list[z_ndx][::self.image_downsample, ::self.image_downsample], autoLevels=auto_levels)
        
        # update imagecount to reflect rendering
        self.image_count = self.shared_image_count[0]
        # self.image_count += 1

    def update_image_LUTs(self):
        # self.lut_histo.setLevels(self.ii_list[0].getLevels())
        # pass
        lev = self.lut_histo.getLevels()
        for z in range(self.zsize):
            self.ii_list[z].setLevels(lev)

    def add_quant_roi_to_image(self, event):

        # read z index from stateful dropdown
        z_ndx = self.quant_roi_combobox.value()
        if not z_ndx: 
            print('please select a z-plane to add roi to')
            return
        z_ndx = int(z_ndx)
        
        # create maleable roi object - note xy is flipped
        roi_id = hash(datetime.now())
        pen_color = self.quant_cmap_list[self.num_rois_added % len(self.quant_cmap_list)]
        quant_roi = pg.RectROI(
            [self.ysize // 2, self.xsize // 2],
            [40, 40],
            pen=pg.mkPen(pen_color, width=2),
        )
        self.num_rois_added += 1

        # store into dict
        roi_dict = {
            'roi_id': roi_id,
            'quant_roi': quant_roi,
            'z_ndx': z_ndx,
            'color': pen_color,
            'roi_dataitem': self.roi_plot_item.plot([], pen=pg.mkPen(pen_color, width=2)),
            # 'roi_deriv_dataitem': self.roi_deriv_plot_item.plot([], pen=pg.mkPen(pen_color, width=2)), 
            'xvals': [],
            'yvals': [],
            'yvals_derivs': [], # necessary for hard deriv calculation, which is based on indexes
        }

        # add roi to list of roi objects, and roi to viewbox
        self.quant_roi_dict_list.append(roi_dict)
        self.image_vbox_list[z_ndx].addItem(quant_roi)

    def add_stim_roi_to_image(self, event):
        
        # read z index from stateful dropdown
        z_ndx = self.stim_roi_combobox.value()
        if not z_ndx: 
            print('please select a z-plane to add roi to')
            return
        z_ndx = int(z_ndx)
        
        # create maleable roi object - note xy is flipped
        roi_id = hash(datetime.now())
        pen_color = self.stim_cmap_list[len(self.stim_roi_dict_list) % len(self.stim_cmap_list)]
        stim_roi = pg.RectROI(
            [self.ysize // 2, self.xsize // 2],
            [40, 40],
            pen=pg.mkPen(pen_color, width=2, style=QtCore.Qt.DotLine),
        )

        # store into dict
        roi_dict = {
            'roi_id': roi_id,
            'stim_roi': stim_roi,
            'z_ndx': z_ndx,
            'color': pen_color,
        }

        # add roi to list of roi objects, and roi to viewbox
        self.stim_roi_dict_list.append(roi_dict)
        self.image_vbox_list[z_ndx].addItem(stim_roi)

    def build_stim_roi_event(self):

        event_data = {}
        event_data = {'x': [], 'y': [], 'width': [], 'height': []}
        for roi_dict in self.stim_roi_dict_list:
            x, y = roi_dict['stim_roi'].pos()
            width, height = roi_dict['stim_roi'].size()
            event_data['x'].append(int(x))
            event_data['y'].append(int(y))
            event_data['width'].append(int(width))
            event_data['height'].append(int(height))

        return event_data
    

    def build_stream_widefield_event(self):
        event_data = {}
        return event_data


    def pulse_stimulus_rois(self, event):

        # if closed loop model is active, skip
        if self.activate_model_button.isChecked(): print('Please deactivate model to pulse stimuli'); return

        # if no roi, skip
        if self.GUI_mode == 'neural_imaging':
            if len(self.stim_roi_dict_list) == 0: print('Please add stimulus rois before trying to stimulate!'); return

            # grab stim rois and format for
            event_type = 'pulse-rect-roi-list'
            event_data = self.build_stim_roi_event()
        
        elif self.GUI_mode == 'behavior':
            event_type = 'stream-widefield'
            event_data = self.build_stream_widefield_event()

        # write event
        self.write_event(event_type, event_data)
    
            
    def pulse_full_field(self, event):

        # if closed loop model is active, skip
        if self.activate_model_button.isChecked(): print('Please deactivate model to pulse stimuli'); return

        self.write_event('full-field-button', None)


    def stream_stimulus_rois(self):

        event_type = 'stream-rect-roi-list'
        event_data = self.build_stim_roi_event()

        if self.GUI_mode == "neural_imaging":
            if len(self.stim_roi_dict_list) == 0: print('Please add stimulus rois before trying to stimulate!'); return

            # grab stim rois and format for
            event_type = 'pulse-rect-roi-list'
            event_data = self.build_stim_roi_event()

        elif self.GUI_mode == 'behavior':
            event_type = 'stream-widefield'
            event_data = self.build_stream_widefield_event()

        # write event
        self.write_event(event_type, event_data)


    def write_event(self, event_type, event_data):

        # don't write events if in refractory period
        if self.stim_refractory_vols > 0:
            return

        data = {
            "ts": time.time(),
            "event_type": event_type,
            "stim_intensity": self.stim_intensity,
        }

        ## write conditional data
        # fixed duration
        if event_type == 'pulse-rect-roi-list' or event_type == "full-field-button":
            data['stim_duration_vols'] = int(self.stim_duration_vols)

        # fixed rois
        if event_type == 'pulse-rect-roi-list' or event_type == 'stream-rect-roi-list':
            data['stim_rect_roi_list'] = event_data

        # write to pipe
        self.child_conn.send(data)

        # visual aid if stim is on
        # fixed duration update refractory counter
        if event_type == "pulse-rect-roi-list" or event_type == 'full-field-button':

            # change button colors and set cooldown
            self.stim_refractory_vols = (
                self.stim_duration_vols + self.refractory_constant
            )
            self.check_stim_refractory_vols()


    def change_stim_buttons_color(self, color):
        self.pulse_stimulus_rois_button.setStyleSheet("background-color: {}".format(self.color_dict[color]))
        self.pulse_full_field_button.setStyleSheet("background-color: {}".format(self.color_dict[color]))
        self.activate_model_button.setStyleSheet("background-color: {}".format(self.color_dict[color]))
        

    def check_stim_refractory_vols(self):
        if self.stim_refractory_vols > 1:
            self.change_stim_buttons_color("red")
        else:
            self.change_stim_buttons_color("green")


    def activate_model_button_callback(self, event):

        # make buttons red
        if self.activate_model_button.isChecked():

            # if no stim rois, reject this
            if self.GUI_mode == "neural_imaging":
                if len(self.stim_roi_dict_list) == 0: 
                    print('Please add stimulus rois before trying to stimulate!')
                    self.activate_model_button.setChecked(False)
                    return

            # make button red
            self.change_stim_buttons_color("red")
            
            # rois can't be moved
            for roi_dict in self.stim_roi_dict_list:
                roi_dict['stim_roi'].translatable = False # undocumented
            for roi_dict in self.quant_roi_dict_list:
                roi_dict['quant_roi'].translatable = False # undocumented
        else: 

            # if stim is currently on, send stop event so alg knows we're shutting auto stim down
            if self.stimulus_is_on:
                self.stream_stimulus_rois()
                self.update_model_output_icon(False)  

            # make button green
            self.change_stim_buttons_color("green")

            # rois can be moved
            for roi_dict in self.stim_roi_dict_list:
                roi_dict['stim_roi'].translatable = True # undocumented
            for roi_dict in self.quant_roi_dict_list:
                roi_dict['quant_roi'].translatable = True # undocumented


    def update_quant_roi_plot(self):

        # get current volume
        curr_vol = self.image_count//self.zsize

        # iterate rois
        for roi_dict in self.quant_roi_dict_list:

            # if we're not updating the right z plane, skip
            roi_z_ndx = roi_dict['z_ndx']
            if self.image_count % self.zsize != roi_z_ndx: continue

            # otherwise grab roi data
            try:
                arr_mean = roi_dict['quant_roi'].getArrayRegion(self.img_list[roi_z_ndx], self.ii_list[roi_z_ndx]).mean()

                # update roi data
                roi_dict['xvals'].append(curr_vol)
                roi_dict['yvals'].append(arr_mean)

                # corner case of first sample, because of deriv calculation
                if len(roi_dict['yvals']) == 1:
                    roi_dict['yvals_derivs'].append(0)
                else:
                    roi_dict['yvals_derivs'].append(roi_dict['yvals'][-1] - roi_dict['yvals'][-2])

                # update plot
                roi_dict['roi_dataitem'].setData(roi_dict['xvals'], roi_dict['yvals'])
                # roi_dict['roi_deriv_dataitem'].setData(roi_dict['xvals'], roi_dict['yvals_derivs'])

            except ValueError as err:
                print('Error while trying to update ROI plot: {}'.format(err))

        # update range of plot
        xstart = max(0, curr_vol - 100)
        self.roi_plot_item.setRange(xRange=[xstart, curr_vol + 1])
        # self.roi_deriv_plot_item.setRange(xRange=[xstart, curr_vol + 1])


    def update_add_delete_quant_roi_button_text(self, event):

        # set button text and image border color
        self.add_quant_roi_button.setText("add quant roi to z-plane: {}".format(self.quant_roi_combobox.value()))
        self.delete_quant_roi_button.setText("delete quant roi from z-plane: {}".format(self.quant_roi_combobox.value()))
        self.update_viewbox_border_color(int(self.quant_roi_combobox.value()))

    def update_add_delete_stim_roi_button_text(self, event):

        # set button text and image border color
        self.add_stim_roi_button.setText("add stim roi to z-plane: {}".format(self.stim_roi_combobox.value()))
        self.delete_stim_roi_button.setText("delete stim roi from z-plane: {}".format(self.stim_roi_combobox.value()))
        self.update_viewbox_border_color(int(self.stim_roi_combobox.value()))

    def update_viewbox_border_color(self, selected_z):

        # add orange border to appropriate z plane
        for z in range(self.zsize):
            if z == selected_z:
                self.image_vbox_list[z].setBorder({'color': self.color_dict['orange'], 'width': 2})
            else:
                self.image_vbox_list[z].setBorder({'color': self.color_dict['grey'], 'width': 2})
            
            # update
            self.image_vbox_list[z].update()


    def delete_quant_roi_from_image(self, event):

        # read z index from stateful dropdown
        z_to_cleanse = int(self.quant_roi_combobox.value())

        # iterate rois on that z-plane
        roi_to_delete = []
        for r in range(len(self.quant_roi_dict_list)):

            # if we're not updating the right z plane, skip
            roi_z_ndx = self.quant_roi_dict_list[r]['z_ndx']
            if z_to_cleanse != roi_z_ndx: continue
            
            # otherwise delete each roi from plot, images, and list
            roi_to_delete.append(self.quant_roi_dict_list[r])

        # now remove all pertinent rois
        for roi_dict in roi_to_delete: 
            self.roi_plot_item.removeItem(roi_dict['roi_dataitem'])
            self.image_vbox_list[roi_z_ndx].removeItem(roi_dict['quant_roi'])
            self.quant_roi_dict_list.remove(roi_dict)


    def delete_stim_roi_from_image(self, event):

        # read z index from stateful dropdown
        z_to_cleanse = int(self.stim_roi_combobox.value())

        # iterate rois on that z-plane
        roi_to_delete = []
        for r in range(len(self.stim_roi_dict_list)):

            # if we're not updating the right z plane, skip
            roi_z_ndx = self.stim_roi_dict_list[r]['z_ndx']
            if z_to_cleanse != roi_z_ndx: continue
            
            # otherwise delete each roi from plot, images, and list
            roi_to_delete.append(self.stim_roi_dict_list[r])

        # now remove all pertinent rois
        for roi_dict in roi_to_delete: 
            self.image_vbox_list[roi_z_ndx].removeItem(roi_dict['stim_roi'])
            self.stim_roi_dict_list.remove(roi_dict)


    def my_get_stimulus_duration_in_vols(self):

        # grab value from user for float
        self.stimulus_duration_vols_dialog = QtWidgets.QInputDialog()

        # paramaterize dialog
        self.stimulus_duration_vols_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)

        # set event handler
        self.stimulus_duration_vols_dialog.intValueSelected.connect(
            self.set_stim_delay_volumes
        )

        # show dialog
        self.stimulus_duration_vols_dialog.show()


    def set_stim_delay_volumes(self, dur):
        self.enter_stim_duration_vols_button.setText(
            "stimulus duration (vols): {}".format(dur)
        )
        self.stim_duration_vols = dur


    def update_model_output_icon(self, status):
        if status:
            self.stimulus_is_on = True
            self.model_output_icon.setText('Model Output: ON')
            self.model_output_icon.setStyleSheet("background-color: {}".format(self.color_dict['orange']))
        else:
            self.stimulus_is_on = False 
            self.model_output_icon.setText('Model Output: OFF')
            self.model_output_icon.setStyleSheet("background-color: {}".format(self.color_dict['grey']))


    def select_model_from_combobox(self, event):
        
        # set internal model pointer
        model_index = self.model_combobox.currentIndex()

        # first index is 'none'
        if model_index == 0:
            self.selected_model = None
            model_name = ''
            model_description = ''
            model_instructions = ''
        else:
            self.selected_model = self.model_list[model_index - 1] 
            model_name = self.selected_model.get_model_name()
            model_description = self.selected_model.get_model_description()
            model_instructions = self.selected_model.get_model_inference_instructions()

        # change model name label
        # self.model_textbox.setText('Model Description: {}\n\nModel Instructions: {}'.format(model_description, model_instructions))
        if self.selected_model is not None:
            
            # self.selected_model.populate_model_panel()

            # remove current model panel
            self.graphics_layout_widget.removeItem(self.model_panel_graphics_layout)

            # set local reference to new model panel
            self.model_panel_graphics_layout = self.selected_model.model_panel_graphics_layout

            # if we're doing behavior model, remove current image renderer
            if self.GUI_mode == "behavior":
                self.graphics_layout_widget.removeItem(self.image_vbox_list[0])
                self.graphics_layout_widget.removeItem(self.lut_histo)
                self.graphics_layout_widget.removeItem(self.vbox_z_label_list[0])
                self.stop_rendering_image = True

            # populate
            row, col, rowspan, colspan = self.get_model_panel_localization_params()
            self.graphics_layout_widget.addItem(self.model_panel_graphics_layout, row=row, col=col, rowspan=rowspan, colspan=colspan)
            self.graphics_layout_widget.update()
            
        else:

            # remove and delete current model panel 
            self.graphics_layout_widget.removeItem(self.model_panel_graphics_layout)
            # self.model_panel_graphics_layout.deleteLater() # if we delete this, then children are also deleted?

            # add new layout and update
            col = (self.zsize+1) // 2
            colspan = int(np.ceil((self.zsize+1) / 2))
            self.model_panel_graphics_layout = self.graphics_layout_widget.addLayout(row=2, col=col, colspan=colspan)
            self.graphics_layout_widget.update()


    def update_model_prediction(self):

        # only update model on full z frames
        if self.image_count % self.zsize != 0:
            return

        # grab model prediction
        if self.selected_model is not None:

            try:
                # make prediction (object has access to all internals)
                pred = self.selected_model.predict()

                # run prediction but don't change output unless approrpriate button is checked
                if self.activate_model_button.isChecked():

                    # if same state, do nothing
                    if not pred and not self.stimulus_is_on:
                        pass
                    elif pred and self.stimulus_is_on:
                        pass

                    # if we're switching states, send an event
                    # turn on
                    elif pred and not self.stimulus_is_on:
                        self.stream_stimulus_rois()
                        self.update_model_output_icon(True)

                    # turn off
                    elif not pred and self.stimulus_is_on:
                        self.stream_stimulus_rois()
                        self.update_model_output_icon(False)


            except Exception as err:
                logging.critical('{}> while trying to make model prediction: {}'.format(self.selected_model.get_model_name(), err))
        
        # if we hit last 10 volumes of recording and stim is on, turn it off, and flip our activate model button
        if self.image_count > self.total_frames - (10*self.zsize):

            # deactivate model toggle, this will turn off stim if it's currently on
            self.activate_model_button.setChecked(False)

# class BrainalyzerImageItem(pg.ImageItem):
#     def __init__(self, *args, **kwargs):
#         super(BrainalyzerImageItem, self).__init__(*args, **kwargs)
#         self.imageitem_id = kwargs.get('imageitem_id', None)

if __name__ == "__main__":

    to_test = 'does autorange viewboxes incur comp cost?'

    # does image downsample give us much? we can update as so
    # self.image_downsample = int(np.ceil(self.image_vbox_list[0].viewPixelSize()[0]))

    # does imageview limits incur comp cost?
    # vb.setLimits(xMin=0, xMax=self.ysize, yMin=0, yMax=self.xsize)
