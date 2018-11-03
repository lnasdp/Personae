# coding=utf-8

import multiprocessing as mp
import pandas as pd
import hashlib
import config
import os

from utility import factor_calculator
from utility.logger import TimeInspector
from base.data_source import BaseDataSource
from spider.stock_spider import StockSpider


class TestDataSource(BaseDataSource):

    def _load_instruments(self):
        self.instruments = ['600001', '600002']

    def _load_origin_data(self):
        columns = ['instrument', 'datetime', 'open', 'close', 'high', 'low', 'volume', 'alpha']
        data = [
            ('600001', '2010-01-01', 3.5, 4.0, 4.1, 3.6, 10000, 1),
            ('600001', '2010-01-02', 4.0, 4.4, 4.6, 3.9, 9000, 0),
            ('600001', '2010-01-03', 4.4, 4.1, 4.5, 3.8, 8000, 1),
            ('600002', '2010-01-02', 4.4, 4.1, 4.5, 3.8, 8000, 0),
            ('600002', '2010-01-04', 4.1, 4.4, 4.6, 3.9, 9000, 0),
            ('600002', '2010-01-05', 3.4, 4.0, 4.1, 3.6, 10000, 1),
            ('600003', '2010-01-04', 4.0, 4.4, 4.6, 3.9, 8000, 1),
            ('600003', '2010-01-02', 4.4, 4.3, 4.5, 3.8, 9000, 0),
            ('600003', '2010-01-05', 4.3, 3.0, 4.1, 3.0, 10000, 1),
        ]

        self.origin_df = pd.DataFrame(data=data, columns=columns)
        self.origin_df['datetime'] = pd.to_datetime(self.origin_df['datetime'])
        self.origin_df = self.origin_df.set_index(['instrument', 'datetime'])
        self.origin_df = self.origin_df.loc(axis=0)[self.instruments]


class TuShareDataSource(BaseDataSource):

    def __init__(self,
                 label_name_selected=config.RETURN_RATE,
                 label_names=None,
                 train_start_date='2017-01-01',
                 train_end_date='2018-01-01',
                 validate_start_date='2018-01-01',
                 validate_end_date='2018-02-01',
                 test_start_date='2018-02-01',
                 test_end_date='2018-03-01',
                 **options):
        super(TuShareDataSource, self).__init__(label_name_selected,
                                                label_names if label_names else [config.RETURN_RATE],
                                                train_start_date,
                                                train_end_date,
                                                validate_start_date,
                                                validate_end_date,
                                                test_start_date,
                                                test_end_date,
                                                **options)

    def _load_instruments(self):
        # 600030 中信证券
        # 600999 招商证券
        # 000166 申万宏源
        # 600837 海通证券
        # 601066 中信建投
        # 002673 西部证券
        self.instruments = config.DEFAULT_INSTRUMENTS

    def _load_origin_data(self):
        self.logger.info('Start load origin data_handler.')
        # Get md5 object.
        md5 = hashlib.md5()
        md5.update(''.join(self.instruments).encode('utf-8'))
        # Get cache data_handler name.
        cache_data_name = md5.hexdigest()
        cache_data_path = os.path.join(config.CACHE_DIR, '{}.pkl'.format(cache_data_name))
        self.logger.info('Cache data_handler name is {}'.format(cache_data_name))
        if not os.path.exists(cache_data_path):
            self.logger.warning('Cache data_handler not exists, read raw data_handler.')
            # If cache data_handler not exist, crawl it.
            raw_df = self.read_raw_data()
            self.origin_df = self.save_origin_data(raw_df, cache_data_name)
        else:
            self.logger.info('Cache data_handler exists, read from cache.')
            self.origin_df = pd.read_pickle(cache_data_path)
            self.logger.info('Finished reading from cache.')

    def read_raw_data(self):
        self.logger.warning('Start reading raw data_handler.')
        TimeInspector.set_time_mark()
        # Concat and cache.
        instrument_frames = []
        for instrument in self.instruments:
            try:
                stock_frame = pd.read_csv(os.path.join(config.STOCK_DATA_DIR, '{}.csv'.format(instrument)),
                                          dtype={'code': str},
                                          parse_dates=['date'])
            except FileNotFoundError:
                raise FileNotFoundError('Stock: {} not found, please run spider first.'.format(instrument))
            instrument_frames.append(stock_frame)
        raw_df = pd.concat(instrument_frames)  # type: pd.DataFrame
        raw_df = raw_df.rename(columns={'date': 'datetime', 'code': 'instrument'})
        raw_df = raw_df.set_index(['instrument', 'datetime'])
        self.logger.warning('Finished reading raw data_handler, time cost: {0:.3f}'.format(TimeInspector.get_cost_time()))
        return raw_df

    def save_origin_data(self, raw_df, cache_data_name):
        self.logger.warning('Start save origin data_handler.')
        TimeInspector.set_time_mark()
        # Cache data_handler path
        if not os.path.exists(config.CACHE_DIR):
            os.makedirs(config.CACHE_DIR)
        raw_df = raw_df.sort_index(level=['instrument', 'datetime'])
        # Calculate factors.
        origin_df = self.calculate_factors(raw_df)
        origin_df.to_pickle(os.path.join(config.CACHE_DIR, '{}.pkl'.format(cache_data_name)))
        self.logger.warning('Finished saving cache data_handler, time cost: {0:.3f}'.format(TimeInspector.get_cost_time()))
        return origin_df

    def calculate_factors(self, origin_df):
        self.logger.warning('Start calculating factors.')
        # 1. Return rate.
        self.logger.warning('Start calculating factor: [{}].'.format(config.RETURN_RATE))
        TimeInspector.set_time_mark()
        origin_df[config.RETURN_RATE] = factor_calculator.calculate_return_rate(origin_df)
        self.logger.warning('Finished calculating factor: [{0}], time cost: {1:.3f}'.format(config.RETURN_RATE, TimeInspector.get_cost_time()))
        # 2. Price diff.
        # self.logger.warning('Start calculating factor: [{}].'.format(config.PRICE_DIFF))
        # TimeInspector.set_time_mark()
        # origin_df[config.PRICE_DIFF] = factor_calculator.calculate_price_diff(origin_df)
        # self.logger.warning('Finished calculating factor: [{0}], time cost: {1:.3f}'.format(config.PRICE_DIFF, TimeInspector.get_cost_time()))
        self.logger.warning('Finished calculating all factors.')
        return origin_df


class RiceQuantDataSource(BaseDataSource):
    pass


if __name__ == '__main__':
    # te = TestDataSource(label_name_selected='alpha', label_names=['alpha'])
    ts = TuShareDataSource()
