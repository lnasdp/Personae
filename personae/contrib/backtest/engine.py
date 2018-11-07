# coding=utf-8

import pandas as pd
import numpy as np
import logging

from abc import abstractmethod

from personae.utility import logger
from personae.utility.profiler import TimeInspector
from personae.contrib.data.loader import PredictorDataLoader


class BaseEngine(object):

    def __init__(self,
                 data_dir,
                 strategy,
                 start_date='2005-01-01',
                 end_date='2018-11-01',
                 cash=1.e8,
                 charge=0.003,
                 slippage=0.05,
                 benchmark='sh000905'):

        # 1. Data.
        self.data_dir = data_dir
        self.data_loader = None
        self.stock_df = None
        self.bench_df = None

        # 2. Strategy.
        self.strategy = strategy

        # 3. Dates.
        self.start_date = start_date
        self.end_date = end_date

        # 4. Trade setting.
        self.cash = cash
        self.charge = charge
        self.slippage = slippage

        # 5. Benchmark.
        self.benchmark = benchmark

        # 6. Setup data.

        TimeInspector.set_time_mark()
        self.setup_data_loader()
        TimeInspector.log_cost_time('Finished setup data loader.')

        TimeInspector.set_time_mark()
        self.setup_stock_data()
        TimeInspector.log_cost_time('Finished setup stock data.')

        TimeInspector.set_time_mark()
        self.setup_bench_data()
        TimeInspector.log_cost_time('Finished setup index data.')

        # 7. Iter dates.
        self.available_dates = self.bench_df.index.get_level_values(level=1).unique().tolist()
        self.iter_dates = iter(self.available_dates)

        # 8. Backtest.
        self.positions = pd.Series(index=self.stock_df.index, data=0)

        # 9. Logger.
        self.logger = logger.get_logger('BACKTEST')

    @abstractmethod
    def setup_data_loader(self):
        raise NotImplementedError

    @abstractmethod
    def setup_stock_data(self):
        raise NotImplementedError

    @abstractmethod
    def setup_bench_data(self):
        raise NotImplementedError

    def run(self):

        # 1. Init cash.
        cash = self.cash
        initial_cash = cash

        # 2. Init dates.
        cur_date = next(self.iter_dates)
        last_date = cur_date

        while True:

            self.strategy.before_trading()

            cur_stock_bar = self.stock_df.loc(axis=0)[:, cur_date]
            cur_stock_close = cur_stock_bar['CLOSE']

            cur_positions = self.strategy.handle_bar(cur_stock_bar)

            last_positions = self.positions.loc(axis=0)[:, last_date]

            positions_diff = cur_positions.reset_index('DATE') - last_positions.reset_index('CODE')

            self.positions.loc(axis=0)[:, cur_date] = cur_positions

            profit = np.sum(last_positions * cur_stock_bar['RETURN_SHIFT_0'])
            loss = np.sum(positions_diff * self.slippage) + np.sum(positions_diff * cur_stock_close * self.charge)

            cash = cash + profit - loss

            self.logger.warning('p is {}, l is {}, cash is {}'.format(profit, loss, cash))

            self.strategy.after_trading()

            last_date = cur_date

            try:
                cur_date = next(self.iter_dates)
            except StopIteration:
                break


class PredictorEngine(BaseEngine):

    def setup_data_loader(self):
        self.data_loader = PredictorDataLoader(self.data_dir, start_date=self.start_date, end_date=self.end_date)

    def setup_stock_data(self):
        self.stock_df = self.data_loader.load_data(codes='all', data_type='stock')

    def setup_bench_data(self):
        self.bench_df = self.data_loader.load_data(codes=[self.benchmark], data_type='index')


if __name__ == '__main__':
    from personae.contrib.strategy.strategy import SampleStrategy
    e = PredictorEngine(r'D:\Users\v-shuyw\data\ycz\data_sample\processed', SampleStrategy())
    e.run()