# coding=utf-8


from abc import abstractmethod


class BaseDataLoader(object):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def load_raw_data(self):
        pass


class PredictorDataLoader(BaseDataLoader):

    def __init__(self, **kwargs):
        super(PredictorDataLoader, self).__init__(**kwargs)

    def load_raw_data(self):
        pass