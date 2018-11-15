# coding=utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

from abc import abstractmethod

from personae.utility import logger
from personae.utility.profiler import TimeInspector


class BaseEngine(object):

    def __init__(self,
                 stock_df,
                 bench_df,
                 start_date='2005-01-01',
                 end_date='2018-11-01',
                 cash=1.e8,
                 charge=0.002,
                 slippage=0.01,
                 benchmark='sh000905',
                 **kwargs):

        # Data df.
        self.stock_df = stock_df.loc(axis=0)[start_date: end_date, :]
        self.bench_df = bench_df.loc(axis=0)[start_date: end_date, :]

        # Strategy.
        self.strategy = None

        # Backtest range.
        self.start_date = start_date
        self.end_date = end_date

        # Benchmark.
        self.benchmark = benchmark

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

        self.cur_positions_weight_dic = dict()
        self.cur_positions_amount_dic = dict()
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
        self.cur_positions_weight_dic[last_date] = pd.Series(index=last_close.index.get_level_values(level='CODE'),
                                                             data=0)
        self.cur_positions_amount_dic[last_date] = pd.Series(index=last_close.index.get_level_values(level='CODE'),
                                                             data=0)

        self.logger.warning('Start backtesting...')

        TimeInspector.set_time_mark()
        while cur_date:

            # Get cur bar.
            try:
                cur_bar = self.stock_df.loc(axis=0)[cur_date]  # type: pd.Series
            except KeyError:
                # Here, all stock cannot trade on this day, we should update positions by last date.
                self.cur_positions_weight_dic[cur_date] = self.cur_positions_weight_dic[last_date]
                self.cur_positions_amount_dic[cur_date] = self.cur_positions_amount_dic[last_date]
                # Update last date and current date.
                last_date, cur_date = cur_date, self.get_next_date()
                # Log and continue.
                self.logger.info('All stock cannot trade on: {}, continue.'.format(cur_date))
                continue

            # Get current positions weight and amount (last date).
            cur_positions_weight = self.cur_positions_weight_dic[last_date]
            cur_positions_amount = self.cur_positions_amount_dic[last_date]

            # Get current available codes.
            cur_codes = cur_bar.index.get_level_values(level='CODE')

            # Call handle bar.
            tar_positions_weight = self.strategy.handle_bar(**{
                'codes': cur_codes,
                'cur_bar': cur_bar,
                'cur_date': cur_date,
                'cur_positions_weight': cur_positions_weight,
            })

            print(self.strategy.alpha)

            """
            Calculate close diff, 
            1. For those codes that not exist on last day , we have current close.
            2. For those codes that not exist on today, we fill forward with last close.
            """
            # Get cur bar return, close.
            cur_close = cur_bar['ADJUST_PRICE']

            # Fill cur close with last close if codes do not exist on today, this cost 10ms.
            close_concat = pd.concat([last_close, cur_close], axis=1, sort=False).fillna(method='ffill', axis=1)
            close_concat.columns = ['LAST', 'CUR']
            cur_close = close_concat.fillna(method='ffill', axis=1).loc(axis=1)['CUR']

            # Calculate close diff.
            close_diff = cur_close.sub(last_close, fill_value=0)

            """
            Calculate positions amount diff, 
            1. For those codes that not exist on last day , we have current amount.
            2. For those codes that not exist on today, we clear with last close.
            """
            # Calculate current holdings values.
            cur_holding_values = np.sum(cur_positions_amount.mul(cur_close, fill_value=0))

            # Calculate current total assets.
            total_assets = cash + cur_holding_values

            # Get target positions amount.
            tar_positions_amount = (tar_positions_weight * total_assets).floordiv(cur_close, fill_value=0)
            # TODO - floor.

            # Calculate positions amount diff.
            positions_amount_diff = tar_positions_amount.sub(cur_positions_amount, fill_value=0)  # type: pd.Series

            # Update current positions weight and amount.
            self.cur_positions_amount_dic[cur_date] = tar_positions_amount
            self.cur_positions_weight_dic[cur_date] = tar_positions_weight

            # Calculate current holdings returns.
            holdings_returns_cash = np.sum(cur_positions_amount * close_diff)

            # Calculate target holdings.
            holdings = np.sum(tar_positions_amount * cur_close)

            # Calculate loss.
            loss_slippage = np.sum(positions_amount_diff.abs() * self.slippage)
            loss_charge = np.sum(positions_amount_diff.abs() * cur_close * self.charge)
            loss = loss_slippage + loss_charge

            # Update total assets.
            total_assets -= loss

            cash = total_assets - holdings

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
                       'Loss: {3:.3f} | ' \
                       'Holdings: {4:.3f} | ' \
                       'RoE: {5:.3f} | ' \
                       'Returns: {6:.6f}'

            self.logger.warning(log_info.format(
                cur_date,
                cash,
                holdings_returns_cash,
                loss,
                holdings,
                roe,
                returns)
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


if __name__ == '__main__':

    from personae.contrib.data.loader import PredictorDataLoader
    from personae.contrib.strategy.strategy import EqualWeightHoldStrategy

    loader = PredictorDataLoader(
        data_dir=r'D:\Users\v-shuyw\data\ycz\data_sample\market_data\data\processed',
        market_dir=r'D:\Users\v-shuyw\data\ycz\data_sample\market_data\market\processed'
    )

    stock_df = loader.load_data('csi500', data_type='stock')
    bench_df = loader.load_data('all', data_type='index')

    e = BaseEngine(
        stock_df=stock_df,
        bench_df=bench_df,
        start_date='2014-01-01',
        end_date='2018-01-01',
        cash=1000000,
        sh_level=logging.INFO)

    e.run(EqualWeightHoldStrategy())
    e.analyze()
    e.plot()
