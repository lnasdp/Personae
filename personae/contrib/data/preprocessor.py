# coding=utf-8

from abc import abstractmethod


class BasePreprocessor(object):

    @abstractmethod
    def process(self):
        pass


class PredictorPreprocessor(BasePreprocessor):

    def __init__(self, **kwargs):
        super(PredictorPreprocessor, self).__init__(**kwargs)

        self.raw_data_dir = kwargs.get('raw_data_dir')
        self.processed_data_dir = kwargs.get('processed_data_dir')
        


