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

        # 1. Data.
        self.processed_data_dir = processed_data_dir
        self.data_loader = None
        self.stock_df = None
        self.bench_df = None

        self.kwargs = kwargs

        # 2. Strategy.
        self.strategy = None

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
        self.available_dates = self.bench_df.index.get_level_values(level=0).unique().tolist()
        self.iter_dates = iter(self.available_dates)

        # 8. Backtest.
        index = pd.MultiIndex.from_product([self.bench_df.index.levels[0].tolist(),
                                            self.stock_df.index.levels[1].tolist()],
                                           names=['DATE', 'CODE'])

        self.positions_se = pd.Series(index=index, data=0).loc[self.start_date: self.end_date, :]
        self.returns_se = pd.Series(index=self.available_dates, data=0.)
        self.roe_se = pd.Series(index=self.available_dates, data=0.)

        # 8. Logger.
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

        self.strategy = strategy

        self.logger.warning('Start backtest.')

        # Cash.
        cash, initial_cash = self.cash, self.cash

        # Dates.
        cur_date = next(self.iter_dates)
        last_date = cur_date

        # Returns.
        stock_returns_se = self.stock_df['RETURN_SHIFT_0']  # type: pd.Series
        stock_returns_se[cur_date] = 0.
        bench_returns_se = self.bench_df['RETURN_SHIFT_0']  # type: pd.Series
        bench_returns_se[cur_date] = 0.

        TimeInspector.set_time_mark()
        while cur_date:

            try:
                cur_stock_bar = self.stock_df.loc(axis=0)[cur_date, :]
            except KeyError:
                # 1. Here, all stock cannot trade on this day, we should update positions.
                self.positions_se.loc[cur_date, :] = self.positions_se.loc(axis=0)[last_date, :]
                # 2. Update last date and current date.
                last_date, cur_date = cur_date, self.get_next_date()
                # 3. Log and continue.
                self.logger.info('All stock cannot trade on: {}, continue.'.format(cur_date))
                continue

            self.strategy.before_trading()

            # 1. Get cur stock bar.
            cur_bar_return = stock_returns_se.loc[cur_date, :].reset_index('DATE', drop=True)
            cur_stock_close = cur_stock_bar['ADJUST_PRICE'].reset_index('DATE', drop=True)

            # 2. Get cur positions.
            cur_positions = self.positions_se.loc(axis=0)[last_date, :].reset_index('DATE', drop=True)

            # 3. Let strategy handle bar.
            tar_positions = self.strategy.handle_bar(cur_stock_bar.reset_index('DATE', drop=True), cur_date)

            if not isinstance(tar_positions, pd.Series):
                raise TypeError('tar_positions should a instance of pd.series.')

            # 4. Update tar positions.
            self.positions_se.loc[cur_date, :] = pd.Series(index=pd.MultiIndex.from_product([[cur_date],
                                                                                             tar_positions.index]),
                                                           data=tar_positions.values)

            # 5. Calculate positions diff.
            positions_diff = tar_positions - cur_positions  # type: pd.Series
            positions_diff = positions_diff.fillna(value=0)

            # 6. Calculate profit.
            profit = np.sum(cur_positions * cur_bar_return)

            # 7. Calculate adjusting holdings cost.
            cost = np.sum(positions_diff * cur_stock_close)

            # 8. Calculate loss.
            loss_slippage = np.sum(positions_diff.abs() * self.slippage)
            loss_charge = np.sum(positions_diff.abs() * cur_stock_close * self.charge)
            loss = -(loss_slippage + loss_charge)

            # 9. Calculate adjusted holdings.
            holdings = np.sum(tar_positions * cur_stock_close)

            # 10. Calculate cash.
            cash += profit
            cash += loss
            cash -= cost

            # 11. Calculate RoE.
            roe = (cash + holdings) / initial_cash

            # 12. Update roe.
            self.roe_se[cur_date] = roe

            # 13. Update returns df.
            returns = roe - self.roe_se[last_date]
            self.returns_se[cur_date] = returns

            log_info = 'Date: {0} | ' \
                       'Profit: {1:.3f} | ' \
                       'Loss: {2:.3f} | ' \
                       'Cost: {3:.3f} | ' \
                       'Cash: {4:.3f} | ' \
                       'Holdings: {5:.3f} | ' \
                       'RoE: {6:.3f} | ' \
                       'Returns: {7:.3f}'
            self.logger.warning(log_info.format(cur_date, profit, loss, cost, cash, holdings, roe, returns))

            self.strategy.after_trading()

            last_date, cur_date = cur_date, self.get_next_date()
        TimeInspector.log_cost_time('Finished backtest.')

    def analyze(self):
        # Portfolio.
        roe = self.roe_se.iloc[-1]
        returns_se = self.returns_se
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
        self.roe_se.plot()
        plt.show()

    def get_next_date(self):
        try:
            cur_date = next(self.iter_dates)
        except StopIteration:
            cur_date = None
        return cur_date


class PredictorEngine(BaseEngine):

    def setup_data_loader(self):
        self.data_loader = PredictorDataLoader(self.processed_data_dir, start_date=self.start_date, end_date=self.end_date)

    def setup_stock_data(self):
        self.stock_df = self.data_loader.load_data(codes='all', data_type='stock')

    def setup_bench_data(self):
        self.bench_df = self.data_loader.load_data(codes=[self.benchmark], data_type='index')


if __name__ == '__main__':
    from personae.contrib.strategy.strategy import HoldStrategy
    e = PredictorEngine(r'D:\Users\v-shuyw\data\ycz\data_sample\processed',
                        start_date='2014-01-01',
                        end_date='2018-01-01',
                        cash=100000,
                        sh_level=logging.INFO)
    e.run(HoldStrategy())
    e.analyze()
    e.plot()
