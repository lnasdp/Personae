# coding=utf-8

import pandas as pd

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
        self.raw_df = None
        self.raw_data_path = kwargs.get('raw_data_path')

    def load_raw_data(self):
        self.raw_df = pd.read_pickle(self.raw_data_path)