# coding=utf-8

from abc import abstractmethod
from utility.logger import get_logger

class BaseDataSource(object):

    def __init__(self,
                 train_start_date='2010-01-01',
                 train_end_date='2010-01-03',
                 validate_start_date='2010-01-03',
                 validate_end_date='2010-01-05',
                 test_start_date='2010-01-05',
                 test_end_date='2010-01-07'):
        # Property.
        self.instruments = None
        self.origin_df = None
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.validate_start_date = validate_start_date
        self.validate_end_date = validate_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.logger = get_logger('DataSource')
        # Method Call.
        self._load_instruments()
        self._load_origin_data()
        # Split df by dates.
        self.train_df = self.origin_df.loc(axis=0)[:, self.train_start_date: self.train_end_date]
        # self.train_df = self.origin_df.loc[:, :]

    @abstractmethod
    def _load_instruments(self):
        pass

    @abstractmethod
    def _load_origin_data(self):
        pass

