class DummyAlg:
    """Class with blank closed-loop algorithm methods"""

    def __init__(self, args=None):
        pass

    def initialize_model(self, args=None):
        pass

    def process_frame(self, frame, zndx):
        pass

    def process_volume(self):
        pass

    def check_stim(self, image_ndx, cooldown_counter):
        return {}, 0

    def get_metadata(self, args=None):
        return {"is_dummy_alg": True}

    def plot_model(self, savefilename=None):
        pass

    def skip(self):
        pass

    def close(self):
        pass
