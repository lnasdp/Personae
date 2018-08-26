# coding=utf-8

import config

from abc import abstractmethod
from utility.logger import get_logger


class BaseDataSource(object):

    def __init__(self,
                 label_name_selected='alpha',
                 label_names=None,
                 train_start_date='2010-01-01',
                 train_end_date='2010-01-03',
                 validate_start_date='2010-01-03',
                 validate_end_date='2010-01-05',
                 test_start_date='2010-01-05',
                 test_end_date='2010-01-07',
                 **options):
        # 1.1 Data related property.
        self.instruments = None
        self.origin_df = None
        self.features_df = None
        self.label_name_selected = label_name_selected
        self.label_names = label_names if label_names else ['alpha']
        self.x_train = None
        self.y_train = None
        self.x_validate = None
        self.y_validate = None
        self.x_test = None
        self.y_test = None
        # 1.2 Date related property.
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.validate_start_date = validate_start_date
        self.validate_end_date = validate_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        # 1.3 Other related property.
        self.logger = get_logger('DataSource')
        # 2. ABS method Call.
        self._load_instruments()
        self._load_origin_data()
        # 3. Process Data related property.
        self.train_df = self.origin_df.loc(axis=0)[:, self.train_start_date: self.train_end_date]
        self.validate_df = self.origin_df.loc(axis=0)[:, self.validate_start_date: self.validate_end_date]
        self.test_df = self.origin_df.loc(axis=0)[:, self.test_start_date: self.test_end_date]
        self.df_columns = self.origin_df.columns.tolist()
        self.feature_columns = list(set(self.df_columns) - set(self.label_names))
        self.features_df = self.origin_df[self.feature_columns]
        # 4. Options property.
        try:
            self.need_normalize_data = options[config.KEY_NEED_NORMALIZE_DATA]
        except KeyError:
            self.need_normalize_data = True
        # 5. Process origin data.
        self.process_origin_data()

    @abstractmethod
    def _load_instruments(self):
        pass

    @abstractmethod
    def _load_origin_data(self):
        pass

    def process_origin_data(self):
        self.x_train = self.train_df[self.feature_columns].values
        self.y_train = self.train_df[[self.label_name_selected]].values
        self.x_validate = self.validate_df[self.feature_columns].values
        self.y_validate = self.validate_df[[self.label_name_selected]].values
        self.x_test = self.test_df[self.feature_columns].values
        self.y_test = self.test_df[[self.label_name_selected]].values
