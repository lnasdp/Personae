# coding=utf-8

import pandas as pd
import numpy as np
import logging
import config

from utility.logger import get_logger


class Backtest(object):

    def __init__(self,
                 data_source,
                 strategy,
                 start_date,
                 end_date,
                 cash=1.e8,
                 slippage=0.05,
                 trade_fee_rate=0.004,
                 sh_level=logging.INFO):
        # 1. Data source.
        self.data_source = data_source
        # 2. Strategy.
        self.strategy = strategy
        # 3. Dates.
        self.start_date = start_date
        self.end_date = end_date
        # 4. Sliced features df.
        self.features_df = self.data_source.features_df.loc(axis=0)[:, self.start_date: self.end_date]
        # 5.1 Check iter dates.
        self.dates = self.features_df.index.get_level_values(level=1).unique().tolist()
        if len(self.dates) < 2:
            raise ValueError('Invalid iter dates length, less than 2 bars.')
        # 5.2 Iter dates.
        self.iter_dates = iter(self.dates)
        self.last_date = pd.Timestamp('1970-01-01')
        self.current_date = next(self.iter_dates)
        self.next_date = next(self.iter_dates)
        # 4. Market setting.
        self.cash = cash
        self.initial_cash = cash
        self.slippage = slippage
        self.trade_fee_rate = trade_fee_rate
        # 5. Positions.
        self.positions = pd.Series(index=self.features_df.index,
                                   data=0)
        # 6. Metrics.
        self.metric_df = pd.DataFrame(index=self.features_df.index.get_level_values(level=1).unique()[:-1],
                                      columns=[config.PROFITS, config.ROE],
                                      data=0,
                                      dtype=float)
        # 7. Others.
        self.logger = get_logger('Backtest', sh_level=sh_level)

    def start(self):
        last_bar_price = self.features_df.loc(axis=0)[:, self.current_date][config.CLOSE]
        last_bar_positions = self.positions.loc(axis=0)[:, self.current_date]
        # Start backtest.
        while True:
            self.logger.debug('On date: {}.'.format(self.current_date))
            # 1. Apply before_trading()
            if self.current_date.day > self.last_date.day:
                self.strategy.before_trading(self.current_date)
                self.logger.debug('`before_trading` called.')
            # 2. Apply handle_bar().
            current_bar_df = self.features_df.loc(axis=0)[:, self.current_date]  # type: pd.DataFrame
            current_bar_positions = self.strategy.handle_bar(current_bar_df, self.current_date)  # type: pd.Series
            self.logger.debug('`handle_bar` called.')
            # 3. Update positions_df.
            self.positions.loc(axis=0)[:, self.current_date] = current_bar_positions
            # 4. Calculate p&l.
            current_bar_price = current_bar_df[config.CLOSE]
            # 4.1 Calculate trade cost.
            positions_diff = current_bar_positions.values - last_bar_positions.values
            trade_cost = (current_bar_price * positions_diff).sum()
            self.logger.debug('Trade cost is {0}'.format(trade_cost))
            # 4.2 Calculate trade fee.
            trade_fee = trade_cost * self.trade_fee_rate + positions_diff.sum() * self.slippage
            self.logger.debug('Trade fee is {0}'.format(trade_fee))
            # 4.3 Calculate profits, here we need values, due to multi-index can not sub directly.
            close_diff = current_bar_price.values - last_bar_price.values
            profits = (close_diff * last_bar_positions).sum()
            self.logger.debug('Profits is {0}'.format(profits))
            # 4.4 Update cash.
            self.logger.debug('Cash before p&l is {0}'.format(self.cash))
            self.cash -= (trade_cost + trade_fee)
            self.cash += profits
            self.logger.debug('Cash after p&l is {0}'.format(self.cash))
            # 4.6 Calculate Holding values.
            holding_values = (current_bar_price * current_bar_positions).sum()
            self.logger.debug('Holding values is {}'.format(holding_values))
            # 4.7 Calculate roe.
            roe = (profits + self.cash + holding_values) / self.initial_cash
            self.logger.debug('RoE is {}'.format(roe))
            # 4.8 Update metric.
            self.metric_df.loc(axis=0)[self.current_date][config.ROE] = roe
            self.metric_df.loc(axis=0)[self.current_date][config.PROFITS] = profits
            # 5. Apply after_trading().
            if self.current_date.day + 1 == self.next_date.day:
                self.strategy.after_trading(self.current_date)
                self.logger.debug('`after_trading` called.')
            # 6. Update dates.
            self.last_date = self.current_date
            self.current_date = self.next_date
            last_bar_price = current_bar_price
            last_bar_positions = current_bar_positions
            try:
                self.next_date = next(self.iter_dates)
            except StopIteration:
                self.logger.warning('Iter dates reached end, backtest over.')
                break

    def analyze(self):
        from matplotlib import pyplot as plt
        self.metric_df[config.ROE].plot()
        plt.show()