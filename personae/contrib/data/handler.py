# coding=utf-8

import pandas as pd

from personae.utility import logger
from personae.utility import profiler

from abc import abstractmethod


class BaseDataHandler(object):

    def __init__(self, **kwargs):

        self.raw_df = None
        self.processed_df = None

        self.label_name = None
        self.label_names = None
        self.feature_names = None

        self.x_train = None
        self.y_train = None

        self.x_validation = None
        self.y_validation = None

        self.x_test = None
        self.y_test = None

        self.w_train = None
        self.w_validation = None

        self.raw_data_path = kwargs.get('raw_data_path')

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

        self.rolling_total_parts = 0
        self.rolling_period = kwargs.get('rolling_period', 30)

        profiler.TimeInspector.set_time_mark()
        self.setup_raw_data()
        profiler.TimeInspector.log_cost_time('Finished loading raw data.')

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
    def setup_raw_data(self):
        pass

    @abstractmethod
    def setup_processed_data(self):
        pass

    @abstractmethod
    def setup_label_names(self):
        pass

    @abstractmethod
    def setup_label(self):
        pass

    @abstractmethod
    def setup_feature_names(self):
        pass

    @abstractmethod
    def setup_static_data(self):
        pass

    @abstractmethod
    def setup_rolling_data(self):
        pass

    @abstractmethod
    def get_split_data_by_dates(self,
                                train_start_date,
                                train_end_date,
                                validate_start_date,
                                validate_end_date,
                                test_start_date,
                                test_end_date):
        pass


class PredictorDataHandler(BaseDataHandler):

    def setup_raw_data(self):
        self.raw_df = pd.read_pickle(self.raw_data_path)
        self.raw_df = self.raw_df.loc(axis=0)[:, '2005-01-01':]

    def setup_processed_data(self):
        # 1. Date slice.
        processed_df = self.raw_df.copy()  # type: pd.DataFrame

        # 2. Drop nan labels.
        profiler.TimeInspector.set_time_mark()
        processed_df = processed_df[~processed_df.loc(axis=1)['return'].isnull()]
        profiler.TimeInspector.log_cost_time('Finished dropping nan labels.')

        self.processed_df = processed_df

    def setup_label_names(self):
        self.label_name = 'return'
        self.label_names = ['return', 'alpha']

    def setup_label(self):
        profiler.TimeInspector.set_time_mark()
        self.processed_df['alpha'] = self.processed_df['return'].groupby(level=1).apply(lambda x: (x - x.mean()) / x.std())
        profiler.TimeInspector.log_cost_time('Finished calculating new label alpha.')

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

        df_train = self.processed_df.loc(axis=0)[:, train_start_date: train_end_date]  # type: pd.DataFrame
        df_validate = self.processed_df.loc(axis=0)[:, validate_start_date: validate_end_date]  # type: pd.DataFrame
        df_test = self.processed_df.loc(axis=0)[:, test_start_date: test_end_date]  # type: pd.DataFrame

        x_train = df_train[self.feature_names].values
        y_train = df_train[self.label_name].values * 100

        x_validate = df_validate[self.feature_names].values
        y_validate = df_validate[self.label_name].values * 100

        x_test = df_test[self.feature_names].values
        y_test = df_test[self.label_name].values * 100

        return x_train, y_train, x_validate, y_validate, x_test, y_test


if __name__ == '__main__':
    h = PredictorDataHandler(raw_data_path=r'D:\Users\v-shuyw\data\ycz\trading-data.20181102\stock_sample\all_market.pkl')
    print(h.x_train)