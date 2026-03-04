class BaseDataInterface:
    def __init__(self):
        """
        Base class for data interfaces.
        """
        pass

    def configure_sampling(self):
        """
        Configure sampling parameters for the data interface.
        """
        pass

    def sample_data(self):
        """
        Retrieve a single data sample from the device.
        """
        pass

    def start_continuous_sampling(self):
        """
        Start continuous data sampling.
        """
        pass

    def stop_continuous_sampling(self):
        """
        Stop continuous data sampling.
        """
        pass

    def get_sample_shape(self):
        """
        Get the shape of data samples.
        """
        pass

    def get_sample_dtype(self):
        """
        Get the data type of samples.
        """
        pass

    def get_metadata(self):
        """
        Get data-specific metadata.
        """
        pass

    def save_data(self):
        """
        Save sampled data to disk or other storage.
        """
        pass

    def close(self):
        """
        Clean up any resources used by the data interface.
        """
        pass