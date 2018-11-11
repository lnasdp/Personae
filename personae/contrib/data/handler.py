# coding=utf-8

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from personae.contrib.data.loader import PredictorDataLoader
from personae.utility import profiler

from abc import abstractmethod


class BaseDataHandler(object):

    def __init__(self, **kwargs):

        # Df.
        self.processed_df = None

        # Labels and features.
        self.label_name = None
        self.label_names = None
        self.feature_names = None

        # Data x, y.
        self.x_train = None
        self.y_train = None

        self.x_validation = None
        self.y_validation = None

        self.x_test = None
        self.y_test = None

        self.w_train = None
        self.w_validation = None

        # Scaler.
        self.scaler = StandardScaler()

        # Raw data dir.
        self.processed_data_dir = kwargs.get('processed_data_dir')

        # Normalize data.
        self.normalize_data = kwargs.get('normalize_data', False)

        # Dates.
        self.train_start_date = kwargs.get('train_start_date', '2016-01-01')
        self.train_end_date = kwargs.get('train_end_date', '2016-12-31')

        self.validation_start_date = kwargs.get('validation_start_date', '2017-01-01')
        self.validation_end_date = kwargs.get('validation_end_date', '2017-12-31')

        self.test_start_date = kwargs.get('test_start_date', '2018-01-01')
        self.test_end_date = kwargs.get('test_end_date', '2018-11-01')

        self.rolling_train_start_dates = []
        self.rolling_train_end_dates = []

        self.rolling_validation_start_dates = []
        self.rolling_validation_end_dates = []

        self.rolling_test_start_dates = []
        self.rolling_test_end_dates = []

        self.rolling_iterator = None
        self.rolling_total_parts = 0
        self.rolling_period = kwargs.get('rolling_period', 30)

        profiler.TimeInspector.set_time_mark()
        self.setup_processed_data()
        profiler.TimeInspector.log_cost_time('Finished loading processing data.')

        profiler.TimeInspector.set_time_mark()
        self.setup_label_names()
        profiler.TimeInspector.log_cost_time('Finished loading label names.')

        profiler.TimeInspector.set_time_mark()
        self.setup_label()
        profiler.TimeInspector.log_cost_time('Finished loading label.')

        profiler.TimeInspector.set_time_mark()
        self.setup_feature_names()
        profiler.TimeInspector.log_cost_time('Finished loading feature names.')

        profiler.TimeInspector.set_time_mark()
        self.setup_static_data()
        profiler.TimeInspector.log_cost_time('Finished loading static data.')

        profiler.TimeInspector.set_time_mark()
        self.setup_rolling_data()
        profiler.TimeInspector.log_cost_time('Finished loading rolling data.')

    def split_rolling_dates(self):
        # 1. Setup rolling time delta.
        rolling_time_delta = pd.Timedelta(value=self.rolling_period, unit='D')
        # 2. Setup cur dates.
        cur_train_start_date = pd.to_datetime(self.train_start_date)
        cur_train_end_date = pd.to_datetime(self.train_end_date)
        cur_validation_start_date = pd.to_datetime(self.validation_start_date)
        cur_validation_end_date = pd.to_datetime(self.validation_end_date)
        cur_test_start_date = pd.to_datetime(self.test_start_date)
        cur_test_end_date = cur_test_start_date + rolling_time_delta - pd.Timedelta(value=1, unit='D')
        # 3. Setup bound date.
        test_bound_date = pd.to_datetime(self.test_end_date)
        # 4. Setup rolling dates.
        self.rolling_train_start_dates = [cur_train_start_date]
        self.rolling_train_end_dates = [cur_train_end_date]
        self.rolling_validation_start_dates = [cur_validation_start_date]
        self.rolling_validation_end_dates = [cur_validation_end_date]
        self.rolling_test_start_dates = [cur_test_start_date]
        self.rolling_test_end_dates = [cur_test_end_date]
        # 5. Split rolling dates.
        need_stop_loop = False
        while not need_stop_loop:
            # 6. Add delta to dates.
            cur_train_start_date += rolling_time_delta
            cur_train_end_date += rolling_time_delta
            cur_validation_start_date += rolling_time_delta
            cur_validation_end_date += rolling_time_delta
            cur_test_start_date += rolling_time_delta
            # 7. Check if reach bound date.
            if cur_test_end_date + rolling_time_delta > test_bound_date:
                rolling_time_delta = test_bound_date - cur_test_end_date
                need_stop_loop = True
            cur_test_end_date += rolling_time_delta
            # 8. Append date to dates.
            self.rolling_train_start_dates.append(cur_train_start_date)
            self.rolling_train_end_dates.append(cur_train_end_date)
            self.rolling_validation_start_dates.append(cur_validation_start_date)
            self.rolling_validation_end_dates.append(cur_validation_end_date)
            self.rolling_test_start_dates.append(cur_test_start_date)
            self.rolling_test_end_dates.append(cur_test_end_date)
        self.rolling_total_parts = len(self.rolling_train_start_dates)

    @abstractmethod
    def setup_processed_data(self):
        raise NotImplementedError('Implement this method to set processed data.')

    @abstractmethod
    def setup_label_names(self):
        raise NotImplementedError('Implement this method to set label names.')

    @abstractmethod
    def setup_label(self):
        raise NotImplementedError('Implement this method to set label.')

    @abstractmethod
    def setup_feature_names(self):
        raise NotImplementedError('Implement this method to set features names.')

    @abstractmethod
    def setup_static_data(self):
        raise NotImplementedError('Implement this method to setup static data.')

    @abstractmethod
    def setup_rolling_data(self):
        raise NotImplementedError('Implement this method to setup rolling data.')

    @abstractmethod
    def get_split_data_by_dates(self,
                                train_start_date,
                                train_end_date,
                                validate_start_date,
                                validate_end_date,
                                test_start_date,
                                test_end_date):
        raise NotImplementedError('Implement this method to get split data.')


class PredictorDataHandler(BaseDataHandler):

    def setup_processed_data(self):

        # Check data dir.
        if not self.processed_data_dir or not os.path.exists(self.processed_data_dir):
            raise ValueError('Invalid raw data dir: {}.'.format(self.processed_data_dir))

        # Here for data handler, the processed data for loader is raw data.
        loader = PredictorDataLoader(self.processed_data_dir, start_date=self.train_start_date, end_date=self.test_end_date)

        # Load processed data.
        processed_df = loader.load_data()  # type: pd.DataFrame

        self.processed_df = processed_df

    def setup_label_names(self):
        self.label_name = 'LABEL_0'
        self.label_names = ['LABEL_0', 'ALPHA']

    def setup_label(self):
        self.processed_df['ALPHA'] = self.processed_df['LABEL_0'].groupby(level=1).apply(lambda x: (x - x.mean()) / x.std())

    def setup_feature_names(self):
        self.feature_names = list(set(self.processed_df.columns) - set(self.label_names))

    def setup_static_data(self):
        split_data = self.get_split_data_by_dates(self.train_start_date,
                                                  self.train_end_date,
                                                  self.validation_start_date,
                                                  self.validation_end_date,
                                                  self.test_start_date,
                                                  self.test_end_date)
        self.x_train = split_data[0]
        self.y_train = split_data[1]

        self.x_validation = split_data[2]
        self.y_validation = split_data[3]

        self.x_test = split_data[4]
        self.y_test = split_data[5]

    def setup_rolling_data(self):
        for index in range(self.rolling_total_parts):
            split_data = self.get_split_data_by_dates(self.rolling_train_start_dates[index],
                                                      self.rolling_train_end_dates[index],
                                                      self.rolling_validation_start_dates[index],
                                                      self.rolling_validation_end_dates[index],
                                                      self.rolling_test_start_dates[index],
                                                      self.rolling_test_end_dates[index])

            self.x_train = split_data[0]
            self.y_train = split_data[1]

            self.x_validation = split_data[2]
            self.y_validation = split_data[3]

            self.x_test = split_data[4]
            self.y_test = split_data[5]

            yield split_data

    def get_split_data_by_dates(self,
                                train_start_date,
                                train_end_date,
                                validate_start_date,
                                validate_end_date,
                                test_start_date,
                                test_end_date):

        df_train = self.processed_df.loc(axis=0)[train_start_date: train_end_date, :]  # type: pd.DataFrame
        df_validate = self.processed_df.loc(axis=0)[validate_start_date: validate_end_date, :]  # type: pd.DataFrame
        df_test = self.processed_df.loc(axis=0)[test_start_date: test_end_date, :]  # type: pd.DataFrame

        x_train = df_train[self.feature_names].values
        y_train = df_train[self.label_name].values * 100

        x_validation = df_validate[self.feature_names].values
        y_validation = df_validate[self.label_name].values * 100

        x_test = df_test[self.feature_names].values
        y_test = df_test[self.label_name].values * 100

        # Normalize data if need.
        if self.normalize_data:
            x_train, x_validation, x_test = self.get_normalized_data(x_train, x_validation, x_test)

        return x_train, y_train, x_validation, y_validation, x_test, y_test

    def get_normalized_data(self, x_train: np.ndarray, x_validation: np.ndarray, x_test: np.ndarray):
        try:
            self.scaler.fit(x_train)
            x_train = self.scaler.transform(x_train)
            x_validation = self.scaler.transform(x_validation)
            x_test = self.scaler.transform(x_test)
        except ValueError as error:
            raise error
        return x_train, x_validation, x_test


if __name__ == '__main__':
    # processed_data_dir = r'D:\Users\v-shuyw\data\ycz\data_sample\processed'
    processed_data_dir = r'D:\Users\v-shuyw\data\ycz\data\processed'
    h = PredictorDataHandler(processed_data_dir=processed_data_dir,
                             normalize_data=True,
                             drop_nan_columns=True)
    print(h.x_train)
