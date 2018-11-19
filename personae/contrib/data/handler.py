# coding=utf-8

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from abc import abstractmethod

from personae.utility import profiler
from personae.utility.logger import get_logger


class BaseDataHandler(object):

    def __init__(self,
                 processed_df,
                 train_start_date='2005-01-01',
                 train_end_date='2014-12-31',
                 validation_start_date='2015-01-01',
                 validation_end_date='2015-06-30',
                 test_start_date='2015-07-01',
                 test_end_date='2017-07-01',
                 **kwargs):

        self.processed_df = processed_df

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

        # Normalize data.
        self.normalize_data = kwargs.get('normalize_data', False)

        # Dates.
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

        self.validation_start_date = validation_start_date
        self.validation_end_date = validation_end_date

        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

        self.rolling_train_start_dates = []
        self.rolling_train_end_dates = []

        self.rolling_validation_start_dates = []
        self.rolling_validation_end_dates = []

        self.rolling_test_start_dates = []
        self.rolling_test_end_dates = []

        self.rolling_iterator = None
        self.rolling_total_parts = 0
        self.rolling_period = kwargs.get('rolling_period', 30)

        self.logger = get_logger('HANDLER')

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
        self.split_rolling_dates()
        self.setup_rolling_data()
        profiler.TimeInspector.log_cost_time('Finished loading rolling data.')

        profiler.TimeInspector.set_time_mark()
        self.check_ic()
        profiler.TimeInspector.log_cost_time('Finished check ic.')

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

    def check_ic(self):
        # Get x space.
        x_space = len(self.x_train.columns)
        # Calculate ic.
        ic = [
            np.corrcoef(self.x_train.values[:, i], self.y_train.values)[0][1] for i in range(x_space)
        ]
        ic = np.abs(np.array(ic))
        # Info.
        info = 'Absolute IC info: Mean: {0:.4f} | Max: {1:.4f} | Min: {2:.4f}'
        # Log.
        self.logger.warning(
            info.format(
                ic.mean(),
                ic.max(),
                ic.min(),
            )
        )

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

    def setup_label_names(self):
        self.label_name = 'LABEL_EWM_ALPHA'
        self.label_names = [
            'LABEL_RETURN',
            'LABEL_ALPHA',
            'LABEL_EWM_RETURN',
            'LABEL_EWM_ALPHA'
        ]

    def setup_label(self):
        pass

    def setup_feature_names(self):
        unused_features = [
            # 'CLOSE',
            # 'ADJUST_PRICE',
            # 'VOLUME',
            'TRADED_MARKET_VALUE',
            'MARKET_VALUE',
            'PE_TTM',
            'PS_TTM',
            'PC_TTM',
            'MONEY',
            'PB'
        ]
        self.feature_names = sorted(list(
            set(self.processed_df.columns) - set(self.label_names) - set(unused_features)
        ))

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

        x_train = df_train[self.feature_names]
        y_train = df_train[self.label_name]

        x_validation = df_validate[self.feature_names]
        y_validation = df_validate[self.label_name]

        x_test = df_test[self.feature_names]
        y_test = df_test[self.label_name]

        # Normalize data if need.
        if self.normalize_data:
            x_train, x_validation, x_test = self.get_normalized_data(x_train, x_validation, x_test)

        return x_train, y_train, x_validation, y_validation, x_test, y_test

    def get_normalized_data(self, x_train, x_validation, x_test):
        try:
            # Fit scaler.
            self.scaler.fit(x_train)
            # Get columns.
            columns = x_train.columns
            # Transform df.
            x_train = pd.DataFrame(
                index=x_train.index,
                columns=columns,
                data=self.scaler.transform(x_train)
            )
            x_validation = pd.DataFrame(
                index=x_validation.index,
                columns=columns,
                data=self.scaler.transform(x_validation)
            )
            x_test = pd.DataFrame(
                index=x_test.index,
                columns=columns,
                data=self.scaler.transform(x_test)
            )
        except ValueError as error:
            raise error
        return x_train, x_validation, x_test
