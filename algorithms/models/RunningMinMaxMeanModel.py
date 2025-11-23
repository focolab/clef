from pyqtgraph.Qt import QtWidgets, QtCore
import sklearn, codecs, pickle, logging
import pyqtgraph as pg
import numpy as np
import json

try:
    from lib.models import BrainalyzerModel # production
except ModuleNotFoundError:
    try:
        from models import BrainalyzerModel # debugging
    except ModuleNotFoundError:
        import BrainalyzerModel # standalone debugging


# sklearn classifiers for the win!!
class RunningMinMaxMeanModel(BrainalyzerModel.BrainalyzerModel):
    def __init__(self, brainalyzer_worker, **kwargs):

        # this model tracks two neurons
        self.num_rois = 2
        self.roi_samples_list = [[] for r in range(self.num_rois)]
        # self.roi_derivs_list = [[] for r in range(self.num_rois)]
        self.sample_xvals = []
        
        # ENUMS
        self.AVA_NDX = 0
        self.SMDV_NDX = 1

        # params
        self.margin = kwargs.get('margin', 0.03)
        self.minmax_window = kwargs.get('minmax_window', 100)
        self.alpha = kwargs.get('alpha', 0.005)
        self.target_quadrant = kwargs.get('target_quadrant', 2)
        
        # internal state vars for model
        self.running_mean = []
        self.running_max = []
        self.running_min = []
        self.running_quadrant = []
        self.running_tstep = []
        self.target_quadrant_list = []

        # holders for initial values
        self.current_max = 0
        self.current_min = 0
        self.initial_estimate = None
        self.last_extrema = 'max'

        # load superconstructor
        super().__init__(brainalyzer_worker, **kwargs)



    def initialize_gui(self):

        # intialize plot object
        self.model_plot_item = pg.PlotItem()
        # self.model_plot_item.disableAutoRange()
        self.model_plot_item.enableAutoRange()
        # current_image_count = self.brainalyzer_worker.image_count
        # self.model_plot_item.setRange(xRange=[current_image_count, current_image_count + 100], yRange=[-20, 20])
        self.model_plot_item.setRange(xRange=[0, 100])
        # self.model_plot_item.setTitle('model output')

        # add a dataitem for smdv
        # pen_color = 'red'
        # pen = pg.mkPen(self.brainalyzer_worker.color_dict[pen_color])
        pen = pg.mkPen("pink")
        self.smdv_dataitem = self.model_plot_item.plot([], pen=pen)
        pen = pg.mkPen('red')
        self.running_max_dataitem = self.model_plot_item.plot([], pen=pen)
        pen = pg.mkPen('blue')
        self.running_min_dataitem = self.model_plot_item.plot([], pen=pen)
        pen = pg.mkPen('white')
        self.running_mean_dataitem = self.model_plot_item.plot([], pen=pen)
        pen = pg.mkPen('orange')
        brush = pg.mkBrush('orange')
        self.quadrant_dataitem = pg.ScatterPlotItem(pen=pen, brush=brush, symbol='s', size=16)
        self.model_plot_item.addItem(self.quadrant_dataitem)

        # create combobox for target quadrant
        self.target_quadrant_combobox = QtWidgets.QComboBox()

        # add quant roi combobox
        self.target_quadrant_combobox = pg.ComboBox(items=[str(x) for x in range(1, 5)])
        self.target_quadrant_combobox.setEditable(True) # for center align 
        self.target_quadrant_combobox_lineEdit = self.target_quadrant_combobox.lineEdit()
        self.target_quadrant_combobox_lineEdit.setAlignment(QtCore.Qt.AlignCenter) # for center align
        self.target_quadrant_combobox_lineEdit.setReadOnly(True)
        self.target_quadrant_combobox.textActivated.connect(self.target_quadrant_combobox_callback)
        self.target_quadrant_combobox.setCurrentIndex(self.target_quadrant - 1) # -1 because 0-indexed

        # wrap in proxy widget
        self.target_quadrant_combobox_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.target_quadrant_combobox_proxy_widget.setWidget(self.target_quadrant_combobox)

        # create spinbox for minmaxmean window size
        self.minmax_window_spinbox = QtWidgets.QSpinBox()
        self.minmax_window_spinbox.setMinimum(10)
        self.minmax_window_spinbox.setMaximum(600)
        self.minmax_window_spinbox.setSingleStep(10)
        self.minmax_window_spinbox.setValue(self.minmax_window)
        self.minmax_window_spinbox.valueChanged.connect(self.minmax_window_spinbox_callback)

        # wrap in proxy widget
        self.minmax_window_spinbox_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.minmax_window_spinbox_proxy_widget.setWidget(self.minmax_window_spinbox)

    def target_quadrant_combobox_callback(self, text):
        self.target_quadrant = self.target_quadrant_combobox.currentIndex() + 1 # +1 because 1-indexed

    def minmax_window_spinbox_callback(self, val):
        self.minmax_window = self.minmax_window_spinbox.value()

    def load_model(self):
        """ load model, context dependent (deserialize, load from file, connect to url etc) """

        pass


    def get_model_dict(self):
        """ save model, context dependent, info necessary to reconstruct model (serialize, save to file, connect to url etc) """

        # try and get some info from brainalyzer_worker
        rec_id = self.brainalyzer_worker.rec_id
        saveroot = self.brainalyzer_worker.saveroot
        
        # nothing pre-saved in this model
        model = {
            'rec_id': rec_id,
            'saveroot': saveroot,
            'margin': self.margin,
            'minmax_window': self.minmax_window,
            'alpha': self.alpha,
            'target_quadrant': self.target_quadrant,
            'running_mean': self.running_mean,
            'running_max': self.running_max,
            'running_min': self.running_min,
            'running_quadrant': self.running_quadrant,
            'running_tstep': self.running_tstep,
            'roi_samples_list': self.roi_samples_list,
        }

        return model


    def store_roi_data(self):

        # store latest roi data values
        rois = self.brainalyzer_worker.quant_roi_dict_list
        
        # nothing happens 
        if len(rois) < self.num_rois:
            logging.warning('Attempting to predict with insufficient number of rois!')
            return

        # grab data from each roi
        # for r in range(self.num_rois):
        for r in [self.AVA_NDX, self.SMDV_NDX]:
            roi = rois[r]
            xval = roi['xvals'][-1]
            # yval = roi['yvals'][-1]
            yval = roi['yvals_derivs'][-1]
            # deriv = roi['yvals_derivs'][-1]

            # add sample to internal tracker
            self.roi_samples_list[r].append(yval)
            # self.roi_derivs_list[r].append(deriv)
        self.sample_xvals.append(xval)

        return True


    def predict(self):
        """ return 0 or 1, 0 for no stim, 1 for stim """

        # grab roi data, if we don't have, return no stim
        result = self.store_roi_data()
        if result is None:
            return 0

        # get latest sample from array
        trace = self.roi_samples_list[self.SMDV_NDX]
        new_val = self.roi_samples_list[self.SMDV_NDX][-1]
        step = len(trace)

        #update current_estimate
        # innovation = new_val - self.current_estimate
        # new_estimate = self.current_estimate + self.alpha * innovation
        # self.current_estimate = new_estimate

        # update current estimate
        if len(self.sample_xvals) < self.minmax_window:
            self.current_max = np.max(trace)
            self.current_min = np.min(trace)
        else:
            self.current_max = np.max(trace[step - self.minmax_window:step])
            self.current_min = np.min(trace[step - self.minmax_window:step])
        self.current_estimate = (self.current_max + self.current_min) / 2

        # udpate current max
        if new_val > (self.current_max - (self.current_estimate * self.margin)):
            self.last_extrema='max'

        # update current min
        if new_val < (self.current_min + (self.current_estimate * self.margin)):
            self.last_extrema='min'

        #compute phase quadrant
        if self.last_extrema=='min':
            if new_val > self.current_estimate:
                self.current_quadrant = 1
            else:
                self.current_quadrant = 4
        else:
            if new_val > self.current_estimate:
                self.current_quadrant = 2
            else:
                self.current_quadrant = 3

        # store vars
        self.running_mean.append(self.current_estimate)
        self.running_max.append(self.current_max)
        self.running_min.append(self.current_min)
        self.running_quadrant.append(self.current_quadrant)
        self.running_tstep.append(self.sample_xvals[-1])
        self.target_quadrant_list.append(self.target_quadrant)

        ###################################################

        # update plot too
        self.update_roi_plot()

        # return whether to stimulate
        if self.current_quadrant == self.target_quadrant:
            return 1
        else:
            return 0


    def update_roi_plot(self):

        # plot SMDV alone
        to_plot = self.roi_samples_list[1]
        self.smdv_dataitem.setData(self.sample_xvals, to_plot)
        self.running_max_dataitem.setData(self.sample_xvals, self.running_max)
        self.running_mean_dataitem.setData(self.sample_xvals, self.running_mean)
        self.running_min_dataitem.setData(self.sample_xvals, self.running_min)

        # update range
        mymin = self.running_min[-1]
        mymax = self.running_max[-1]
        # self.model_plot_item.setRange(xRange=(self.sample_xvals[-self.minmax_window], self.sample_xvals[-1]), yRange=(mymin, mymax))
        x0 = min(len(self.sample_xvals), self.minmax_window)
        self.model_plot_item.setRange(xRange=(self.sample_xvals[-x0], self.sample_xvals[-1]), yRange=(mymin, mymax))

        # get quadrant labeling in range and render
        quad_data = np.array(self.running_quadrant[-self.minmax_window:])
        quadrant_hits = np.where(quad_data == self.target_quadrant)[0]
        xs = np.array(self.sample_xvals[-self.minmax_window:])[quadrant_hits]
        ys = np.ones_like(xs) * mymin
        self.quadrant_dataitem.setData(x=xs, y=ys)
        

    def create_model_panel(self):

        # intialize (but don't add!) our gui objects
        self.initialize_gui()

        # create graphics layout, might have to set parent to brainalyzerworker's graphicslayoutwidget
        # self.model_panel_graphics_layout = pg.GraphicsLayout() # done in constructor

        # add label
        self.model_panel_graphics_layout.addLabel('Model Description: {}'.format(self.get_model_description()), row=0, col=0, colspan=6)
        self.model_panel_graphics_layout.addLabel('Model Instructions: {}'.format(self.get_model_inference_instructions()), row=1, col=0, colspan=6)

        # add plot below
        model_plot_item_rowspan = 12
        self.model_panel_graphics_layout.addItem(self.model_plot_item, row=2, rowspan=model_plot_item_rowspan, col=0, colspan=6)

        # add combobox for target quadrant
        self.model_panel_graphics_layout.addLabel('Target Quadrant:', row=0, col=7, rowspan=1)
        self.model_panel_graphics_layout.addItem(self.target_quadrant_combobox_proxy_widget, row=1, col=7, colspan=1)

        # add spinbox for minmax window
        self.model_panel_graphics_layout.addLabel('Window Size:', row=2, col=7, rowspan=1)
        self.model_panel_graphics_layout.addItem(self.minmax_window_spinbox_proxy_widget, row=3, col=7, colspan=1)

        # add spacers to get labels and objects the right scaling
        r = 4
        while r <= model_plot_item_rowspan + 1:
            self.model_panel_graphics_layout.addLabel('', row=r, col=7, rowspan=1)
            r += 1


    def save_model(self):

        # try and save model info for later debugging
        try:
            model_dict = self.get_model_dict()
            saveroot = model_dict.get('saveroot', '')
            model_output_fname = saveroot + '_RunningMinMaxMeanModel_output.json'

            with open(model_output_fname, 'w') as f:
                json.dump(model_dict, f, indent=4, sort_keys=True)
            logging.info('Saved model output to: {}'.format(model_output_fname))
            
        except Exception as err:
            logging.warning('Failed to save model info! {}'.format(err))
        

    def close(self):
        self.save_model()


## for debugging
# advance state function to give to debugger to advance debugger's state
# necessary for algs which assume brainalyzer e.g. has rois that it's updating
def advance_state(self):

    # grab data from internal data structure 
    # note this could have off-by-one issue cause of when image count is incremented... will need to think about timing
    xval = self.state_count
    yval_smdv = np.mean(self.data_struct_smdv[self.state_count]) # just take mean of mini image
    yval_ava = np.mean(self.data_struct_ava[self.state_count]) # just take mean of mini image

    # add an object mimic'ing the object present in brainalyzer_worker
    self.quant_roi_dict_list[0]['xvals'].append(xval)
    self.quant_roi_dict_list[0]['yvals'].append(yval_ava)
    self.quant_roi_dict_list[1]['xvals'].append(xval)
    self.quant_roi_dict_list[1]['yvals'].append(yval_smdv)


# add necessary internal state vars to debugger, and set advance_state function
def add_debugging_statevars(bug, neuron_tcrop_rec_dict):

    data_struct_smdv = neuron_tcrop_rec_dict['SMDV']
    data_struct_ava = neuron_tcrop_rec_dict['AVA']

    bug.data_struct_smdv = data_struct_smdv # sets an internal variable which will be referenced by advance_state
    bug.data_struct_ava = data_struct_ava
    bug.quant_roi_dict_list = [
        {
            'roi': None,
            'xvals': [],
            'yvals': [],
        },
        {
            'roi': None,
            'xvals': [],
            'yvals': [],
        }
    ]
    bug.set_advance_state_function(advance_state)


# standalone testing
if __name__ == '__main__':

    # # params
    num_t = 500
    
    # # load data we're simulating
    # ds_fname = 'C:/Users/rldun/data/20230226_neuron_tcrop_dict/neuron_tcrop_dict.pkl'
    ds_fname = 'C:/Users/rldun/code/wb-live/analysis/notebooks/20230519 analysis/modeling/data/neuron_tcrop_dict.pkl'
    
    # load data and add to debugger
    rec = '20230322-21-41-10'
    neuron_tcrop_dict = pickle.load(open(ds_fname, 'rb'))

    # model arguments
    model_args = {
        "model_name": "test"
    }

    # debugger arguments
    debugger_args = {
        'timestep_seconds': 0.05,
        'rec_id': 'test',
        'saveroot': 'C:/Users/rldun/code/wb-live/analysis/notebooks/20230519 analysis/modeling/data'
    }


    ##############################################################################################

    # imports
    import BrainalyzerModelDebugger

    # instantiate debugger object
    bug = BrainalyzerModelDebugger.BrainalyzerModelDebugger(debugger_args=debugger_args)

    # define advance_state function (+ necessary internal vars) and, add to debugger
    add_debugging_statevars(bug, neuron_tcrop_dict[rec])

    # instantiate model object with debugger instead of brainalyzer_worker as object
    json_fname = 'running_min_max_mean_model.json'
    mod = RunningMinMaxMeanModel.from_json(json_fname, bug, **model_args)

    # link model panel to debugger
    bug.link_model_panel(mod)

    # iterate states
    print('Running MinMaxMeanModel')
    for i in range(0, num_t):

        # advance simulation state and advance prediction
        bug.advance_state()
        mod.predict()

        # iterate frames "states"
        if i % 100 == 0:
            print(i)

    bug.close()
    mod.close()
