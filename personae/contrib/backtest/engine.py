# coding=utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

from abc import abstractmethod

from personae.utility import logger
from personae.utility.profiler import TimeInspector
from personae.contrib.data.loader import PredictorDataLoader


class BaseEngine(object):

    def __init__(self,
                 processed_data_dir,
                 start_date='2005-01-01',
                 end_date='2018-11-01',
                 cash=1.e8,
                 charge=0.002,
                 slippage=0.01,
                 benchmark='sh000905',
                 **kwargs):

        # Data dir.
        self.processed_data_dir = processed_data_dir

        # Data loader.
        self.data_loader = None

        # Data df.
        self.stock_df = None
        self.bench_df = None

        # Strategy.
        self.strategy = None

        # Backtest range.
        self.start_date = start_date
        self.end_date = end_date

        # Benchmark.
        self.benchmark = benchmark

        # Setup data.
        TimeInspector.set_time_mark()
        self.setup_data_loader()
        TimeInspector.log_cost_time('Finished setup data loader.')

        TimeInspector.set_time_mark()
        self.setup_stock_data()
        TimeInspector.log_cost_time('Finished setup stock data.')

        TimeInspector.set_time_mark()
        self.setup_bench_data()
        TimeInspector.log_cost_time('Finished setup index data.')

        # Backtest codes.
        self.available_codes = self.stock_df.index.get_level_values(level=1).unique().tolist()
        self.available_dates = self.bench_df.index.get_level_values(level=0).unique().tolist()

        if len(self.available_codes) < 1:
            raise ValueError('Available codes less than 1, please check data.')

        if len(self.available_dates) < 2:
            raise ValueError('Available dates less than 3, please check data.')

        # Backtest dates.
        self.iter_dates = iter(self.available_dates)
        self.backtest_end_date = None

        # Backtest trade info.
        self.cash = cash
        self.charge = charge
        self.slippage = slippage

        self.cur_positions_dic = dict()
        self.return_dic = dict()
        self.roe_dic = dict()

        # Logger.
        self.logger = logger.get_logger('BACKTEST', **kwargs)

    @abstractmethod
    def setup_data_loader(self):
        raise NotImplementedError('Implement this method to setup data loader.')

    @abstractmethod
    def setup_stock_data(self):
        raise NotImplementedError('Implement this method to setup stock data')

    @abstractmethod
    def setup_bench_data(self):
        raise NotImplementedError('Implement this method to setup bench data')

    def run(self, strategy):

        # Strategy.
        self.strategy = strategy

        # Iter dates.
        self.iter_dates = iter(self.available_dates)

        # Cash.
        cash, initial_cash = self.cash, self.cash

        # Cur date.
        cur_date = next(self.iter_dates)

        # Last date.
        last_date = cur_date

        # Last close
        last_close = self.stock_df.loc(axis=0)[last_date]['ADJUST_PRICE']

        # Positions.
        self.cur_positions_dic[last_date] = pd.Series(index=self.available_codes, data=0)

        self.logger.warning('Start backtesting...')

        TimeInspector.set_time_mark()
        while cur_date:

            # Get cur bar.
            try:
                cur_bar = self.stock_df.loc(axis=0)[cur_date]
            except KeyError:
                # Here, all stock cannot trade on this day, we should update positions by last date.
                self.cur_positions_dic[cur_date] = self.cur_positions_dic[last_date]
                # Update last date and current date.
                last_date, cur_date = cur_date, self.get_next_date()
                # Log and continue.
                self.logger.info('All stock cannot trade on: {}, continue.'.format(cur_date))
                continue

            # Call handle bar.
            tar_positions = self.strategy.handle_bar(**{
                'cur_date': cur_date,
                'codes': self.available_codes,
            })

            # Get cur bar return, close.
            cur_close = cur_bar['ADJUST_PRICE']

            # Get current positions (last date).
            cur_positions = self.cur_positions_dic[last_date]

            # Update current positions.
            self.cur_positions_dic[cur_date] = tar_positions

            # Calculate positions diff.
            positions_diff = tar_positions.sub(cur_positions, fill_value=0)  # type: pd.Series

            # Calculate close diff.
            close_diff = cur_close.sub(last_close, fill_value=0)

            # Calculate current holdings returns.
            holdings_return = np.sum(cur_positions * close_diff)

            # Calculate adjusting cash.
            holdings_adjusted_return = np.sum((-positions_diff * cur_close))

            # Calculate holdings.
            holdings = np.sum(tar_positions * cur_close)

            # Calculate loss.
            loss_slippage = np.sum(positions_diff.abs() * self.slippage)
            loss_charge = np.sum(positions_diff.abs() * cur_close * self.charge)
            loss = loss_slippage + loss_charge

            loss = 0

            # Calculate cash.
            cash -= loss
            cash += holdings_adjusted_return
            cash += holdings_return

            # Calculate RoE.
            roe = (cash + holdings) / initial_cash

            # Update roe.
            self.roe_dic[cur_date] = roe

            # Update return.
            returns = roe - self.roe_dic[last_date]
            self.return_dic[cur_date] = returns

            log_info = 'Date: {0} | ' \
                       'Cash: {1:.3f} | ' \
                       'Profit: {2:.3f} | ' \
                       'Holdings: {3:.3f} | ' \
                       'RoE: {4:.3f} | ' \
                       'Returns: {5:.6f}'

            self.logger.warning(log_info.format(
                cur_date,
                cash,
                holdings_return, #holdings_adjusted_return +
                holdings,
                roe, returns*initial_cash)
            )

            last_date, cur_date, last_close = cur_date, self.get_next_date(), cur_close
        # Update backtest end date.
        self.backtest_end_date = last_date
        TimeInspector.log_cost_time('Finished backtest.')

    def analyze(self):
        # Portfolio.
        roe = self.roe_dic[self.backtest_end_date]
        returns_se = pd.Series(self.return_dic)
        returns_mean = returns_se.mean()
        returns_std = returns_se.std()
        annual = returns_mean * 250
        sharpe = returns_mean / returns_std * np.sqrt(250)
        mdd = ((returns_se.cumsum() - returns_se.cumsum().cummax()) / (1 + returns_se.cumsum().cummax())).min()
        performance = pd.Series({
            'annual': annual,
            'sharpe': sharpe,
            'mdd': mdd,
            'roe': roe
        })
        self.logger.warning('\n{}\n'.format(performance))
        return performance

    def plot(self):
        plt.figure()
        pd.Series(self.roe_dic).plot()
        plt.show()

    def get_next_date(self):
        try:
            next_date = next(self.iter_dates)
        except StopIteration:
            next_date = None
        return next_date


class PredictorEngine(BaseEngine):

    def setup_data_loader(self):
        self.data_loader = PredictorDataLoader(self.processed_data_dir, start_date=self.start_date, end_date=self.end_date)

    def setup_stock_data(self):
        self.stock_df = self.data_loader.load_data(codes='all', data_type='stock')
        self.stock_df['ALPHA'] = self.stock_df['LABEL_0'].groupby(level=0).apply(
            lambda x: (x - x.mean()) / x.std()
        )

    def setup_bench_data(self):
        self.bench_df = self.data_loader.load_data(codes=[self.benchmark], data_type='index')


if __name__ == '__main__':
    from personae.contrib.strategy.strategy import HoldStrategy
    e = PredictorEngine(r'D:\Users\v-shuyw\data\ycz\data_sample\processed',
                        start_date='2014-01-01',
                        end_date='2018-01-01',
                        cash=1000000,
                        sh_level=logging.INFO)
    e.run(HoldStrategy())
    e.analyze()
    e.plot()
