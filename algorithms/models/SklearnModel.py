import sklearn, codecs, pickle

try:
    from lib.models import BrainalyzerModel
except ModuleNotFoundError:
    from models import BrainalyzerModel


# sklearn classifiers for the win!!
class SklearnModel(BrainalyzerModel.BrainalyzerModel):
    def __init__(self, brainalyzer_worker, **kwargs):

        # load superconstructor
        super().__init__(brainalyzer_worker, **kwargs)


    def load_model(self):
        """ load model, context dependent (deserialize, load from file, connect to url etc) """

        # in this case "model" is python dict, including a serialized sklearn model, encoded in base64
        # initialization it's still be 
        self.clf = lazy_serialize(self.model['clf'], to_str=False)


    def save_model(self):
        """ save model, context dependent, info necessary to reconstruct model (serialize, save to file, connect to url etc) """
        
        # in this case "model" is python dict, including a serialized sklearn model
        # json doesn't take bytes so we have to serialize model and encode in b64
        # this is dangerous if applied to untrusted bytes
        b64 = lazy_serialize(self.clf, to_str=True)

        model = {
            'clf': b64
        }
        return model
        

    def predict(self):
        """ return 0 or 1, 0 for no stim, 1 for stim """

        # # grab data from rois
        # model_window_size = 1
        # data = []
        # for roi_dict in self.brainalyzer_worker.quant_roi_dict_list:
        #     samples = roi_dict['yvals'][-model_window_size:]
        #     if len(samples) < model_window_size:
        #         raise(Exception('Not enough samples to make model prediction'))
        #     data.append(samples)



        # # grab some recent timehistory data
        # x1 = roi_1['yvals'][:5]
        # x2 = roi_2['yvals'][:5]

        # # make a prediction, binary classification stim or not stim
        # pred = self.clf.predict([x1, x2])
        pred = ((self.brainalyzer_worker.image_count // self.brainalyzer_worker.zsize) // 25) % 2

        # print('image count: {}, prediction: {}'.format(self.brainalyzer_worker.image_count, pred))
        return pred


# serialization/deserialization scheme for storing pickled objects in json files
def lazy_serialize(res, to_str=True):

    # if we want to convert to a string
    if to_str:

        # serialize into bytes
        b = pickle.dumps(res)

        # encode bytes as base64 bytes
        b64 = codecs.encode(b, "base64")

        # convert to string
        return b64.decode()

    # if we want to take a string and convert it to an obj
    else:

        # take json-loaded string and encode as b64
        b64 = codecs.decode(res.encode(), "base64")

        # load b64 as python obj
        return pickle.loads(b64)