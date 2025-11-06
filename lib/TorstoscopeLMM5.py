import StimBaseClass

print('Not done! Needs to be refactored to match InvCoreLDIPolygon e.g. use pycromanager')

class TorstoscopeLMM5(StimBaseClass):
    def __init__(self, args, local_handles={})

        # call superconstructor
        super().__init__(args, local_handles)

        # load handle to microscope hardware
        self.mmc = self.get_mmc()
        