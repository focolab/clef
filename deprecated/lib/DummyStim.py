from lib import StimBaseClass


class DummyStim(StimBaseClass.StimBaseClass):

    def __init__(self, args, **kwargs):

        # superconstructer
        super().__init__(args)

    def submit_stim_params(self, stim_params, image_ndx):
        pass

    def activate_stim(self):
        pass

    def inactivate_stim(self):
        pass



