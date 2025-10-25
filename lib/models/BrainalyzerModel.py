from pyqtgraph.Qt import QtWidgets, QtGui
from abc import ABC, abstractmethod
import json
import pyqtgraph as pg


class BrainalyzerModel(ABC):

    def __init__(self, brainalyzer_worker, **kwargs):

        # some params for gui
        self.brainalyzer_worker = brainalyzer_worker
        self.model_name = kwargs.get('model_name', 'test')
        self.model_class = kwargs.get('model_class', 'test')
        self.model = kwargs.get('model', None)
        self.model_description = kwargs.get('model_description', '')
        self.model_inference_instructions = kwargs.get('model_inference_instructions', '')

        # model card, per https://arxiv.org/pdf/1810.03993.pdf
        # note sklearn mentions https://dmg.org/pmml/v4-4-1/GeneralStructure.html for heavyweight spec, but it's not clear if it's widely used 
        # lets load other model_card features but not necessarily do anything with them, for min spec
        model_card = kwargs.get('model_card', {})
        self.model_details = model_card.get('model_details', {})
        self.intended_use = model_card.get('intended_use', {})
        self.factors = model_card.get('factors', {})
        self.metrics = model_card.get('metrics', {})
        self.evaluation_data = model_card.get('evaluation_data', {})
        self.training_data = model_card.get('training_data', {})
        self.quantitative_analyses = model_card.get('quantitative_analyses', {})
        self.ethical_considerations = model_card.get('ethical_considerations', {})
        self.caveats_and_recommendations = model_card.get('caveats_and_recommendations', {})

        # load model
        self.load_model()

        # models get a graphics layout for displaying relevant info and interactable objects
        self.model_panel_graphics_layout = pg.GraphicsLayout()
        self.create_model_panel()

    @abstractmethod
    def load_model(self):
        """ load model, context dependent (deserialize, load from file, connect to url etc) """
        pass

    @abstractmethod
    def save_model(self):
        """ save model, context dependent, info necessary to reconstruct model (serialize, save to file, connect to url etc) """
        pass

    @abstractmethod
    def predict(self):
        """ inference """ 
        pass

    def get_model_name(self):
        return self.model_name

    def get_model_description(self):
        return self.model_description

    def get_model_window_size(self):
        return self.model_time_history_window

    def get_model_inference_instructions(self):
        return self.model_inference_instructions

    def get_model_panel(self):
        """ return a viewbox for displaying model output """
        return self.brainalyzer_worker.model_viewbox

    def to_json(self, save_to_file=False, json_fname='model.json'):

        # create a dict of the model and save to json
        model_dict = {
            'model_name': self.model_name,
            'model': self.save_model(),
            'model_description': self.model_description,
            'model_inference_instructions': self.model_inference_instructions,
            'model_card': {
                'model_details': self.model_details,
                'intended_use': self.intended_use,
                'factors': self.factors,
                'metrics': self.metrics,
                'evaluation_data': self.evaluation_data,
                'training_data': self.training_data,
                'quantitative_analyses': self.quantitative_analyses,
                'ethical_considerations': self.ethical_considerations,
                'caveats_and_recommendations': self.caveats_and_recommendations,
            }
        }

        # return or save to file
        if save_to_file:
            with open(json_fname, 'w') as f:
                json.dump(model_dict, f)
        else:
            return model_dict


    def create_model_panel(self):

        # create simple textbox widget
        self.model_textbox = QtWidgets.QTextEdit()
        self.model_textbox.setText('Model Description: {}\n\nModel Instructions: {}'.format(self.get_model_description(), self.get_model_inference_instructions()))
        self.model_textbox.setReadOnly(True)
        self.model_textbox.setWordWrapMode(QtGui.QTextOption.WordWrap)
        self.model_textbox.setAutoFillBackground(False)

        # graphics layouts require graphics widgets. to add regular widgets we wrap in proxy widget
        self.model_panel_proxy_widget = QtWidgets.QGraphicsProxyWidget()
        self.model_panel_proxy_widget.setWidget(self.model_textbox)
        self.model_panel_graphics_layout.addItem(self.model_panel_proxy_widget)

    # breakdown 
    def close(self):
        pass


    @staticmethod
    def from_json(json_fname, brainalyzer_worker, **kwargs):
        """ load a brainalyzer model from a json file """

        with open(json_fname, 'r') as f:

            # load the json file
            json_data = json.load(f)

            # based on model type, load the appropriate subclass
            model_class = json_data.get('model_class', None)
            if model_class == 'sklearn':
                try:
                    from lib.models import SklearnModel # deployment
                except ModuleNotFoundError:
                    from models import SklearnModel # for local testing
                    
                bm = SklearnModel.SklearnModel(brainalyzer_worker, **json_data)

            elif model_class == 'running_min_max_mean_model':

                try:
                    from lib.models import RunningMinMaxMeanModel # deployment
                except ModuleNotFoundError:
                    try:
                        from models import RunningMinMaxMeanModel # testing
                    except ModuleNotFoundError:
                        import RunningMinMaxMeanModel # standalone testing

                bm = RunningMinMaxMeanModel.RunningMinMaxMeanModel(brainalyzer_worker, **json_data)

            elif model_class == "single_worm_tracker":
                try:
                    from lib.models import HeadCurvatureAndXYStageModel
                except ModuleNotFoundError:
                    try:
                        from models import HeadCurvatureAndXYStageModel
                    except ModuleNotFoundError:
                        import HeadCurvatureAndXYStageModel
                
                bm = HeadCurvatureAndXYStageModel.HeadCurvatureAndXYStageModel(brainalyzer_worker, **json_data)


        return bm

