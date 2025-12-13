import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import time


class BrainalyzerModelDebugger:

    # on initialization create gui
    def __init__(self, debugger_args: dict={}):

        # unpack some args
        self.timestep_seconds = debugger_args.get('timestep_seconds', 1)
        self.rec_id = debugger_args.get('rec_id', 'test')
        self.saveroot = debugger_args.get('saveroot', '')

        # set some internal refs
        self.state_count = 0
        self.color_dict = {}

        # instantiate gui
        self.instantiate_gui()


    def instantiate_gui(self):

        # instantiate app
        self.app = pg.Qt.mkQApp(name='BrainalyzerModelDebugger')

        # make text label keeping track of state advances
        self.state_count_qlabel = QtWidgets.QLabel("state: {}".format(self.state_count))
        self.text_layout = QtWidgets.QGridLayout()
        self.text_layout.addWidget(self.state_count_qlabel, 0, 0)

        # create layout and add it. model can then update that layout with plots and widgets (via proxy)
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.model_panel_graphics_layout = self.graphics_layout_widget.addLayout()

        # add graphics layout to grid layout
        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.addWidget(self.graphics_layout_widget, 0, 0)
        self.grid_layout.addItem(self.text_layout, 1, 0)

        # add grid layout to window and show
        self.window = QtWidgets.QWidget()
        self.window.resize(1400, 800)
        self.window.setLayout(self.grid_layout)
        self.window.show()

        # run app
        # self.app.exec()


    def link_model_panel(self, brainalyzer_model):

        # remove current model panel
        self.graphics_layout_widget.removeItem(self.model_panel_graphics_layout)

        # set local reference to new model panel
        self.model_panel_graphics_layout = brainalyzer_model.model_panel_graphics_layout

        # link and update
        self.graphics_layout_widget.addItem(self.model_panel_graphics_layout)
        self.graphics_layout_widget.update()


    # externally, the "advance_state_function" is provided by whatever model. set internal ref to this function
    def set_advance_state_function(self, advance_state_function):
        self.advance_state_ = advance_state_function


    def update_state_count_text(self):
        self.state_count_qlabel.setText("state: {}".format(self.state_count))


    # function to advance the state of our simulator
    def advance_state(self):

        # advance the state of the debugger
        self.advance_state_(self)
        self.state_count += 1
        self.update_state_count_text()
        self.app.processEvents()

        # run event loop while waiting
        tstart = time.time()
        while time.time() < tstart + self.timestep_seconds:
            self.app.processEvents()


    def close(self):
        pass

    
    