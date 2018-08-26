# coding=utf-8

import pandas as pd
import logging
import config

from utility.logger import get_logger


class Backtest(object):

    def __init__(self,
                 data_source,
                 strategy,
                 start_date,
                 end_date,
                 cash=1e8,
                 slippage=0.4,
                 trade_fee_rate=0.004,
                 verbose=True):
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
        self.dates = self.features_df.index.levels[1].unique().tolist()
        if len(self.dates) < 2:
            raise ValueError('Invalid iter dates length, less than 2 bars.')
        # 5.2 Iter dates.
        self.iter_dates = iter(self.dates)
        self.last_date = pd.Timestamp('1970-01-01')
        self.current_date = next(self.iter_dates)
        self.next_date = next(self.iter_dates)
        # 4. Market setting.
        self.cash = cash
        self.slippage = slippage
        self.trade_fee_rate = trade_fee_rate
        # 5. Positions.
        self.positions_series = pd.Series(index=self.features_df.index, data=0)
        # 6. Metrics.
        self.metric_df = pd.DataFrame(index=self.features_df.index.levels[1], columns=['sharpe'], data=0)
        # 7. Others.
        self.logger = get_logger('Backtest', sh_level=logging.WARNING if not verbose else logging.INFO)

    def start(self):
        last_bar_close = None
        last_bar_positions_series = None
        # Start backtest.
        while True:
            # 1. Apply before_trading()
            if self.current_date.day > self.last_date.day:
                self.strategy.before_trading(self.current_date)
                self.logger.info('`before_trading` called on: {}'.format(self.current_date))
            # 2. Apply handle_bar().
            bar_df = self.features_df.loc(axis=0)[:, self.current_date]  # type: pd.DataFrame
            bar_positions_series = self.strategy.handle_bar(bar_df, self.current_date)  # type: pd.Series
            self.logger.info('`handle_bar` called on: {}'.format(self.current_date))
            # 3. Update positions_df.
            self.positions_series.loc(axis=0)[:, self.current_date] = bar_positions_series
            # 4. Update metric.
            bar_close = bar_df['close']
            # 4.1 Calculate trade cost.
            trade_cost = bar_close * bar_positions_series
            # 4.2 Calculate trade fee.
            trade_fee = trade_cost * self.trade_fee_rate + bar_positions_series.sum() * self.slippage
            # 4.3 Calculate profits.
            if last_bar_close and last_bar_positions_series:
                close_diff = bar_close - last_bar_close
                positions_diff = bar_positions_series - last_bar_positions_series


            # 4. Apply after_trading().
            if self.current_date.day + 1 == self.next_date.day:
                self.strategy.after_trading(self.current_date)
                self.logger.info('`after_trading` called on: {}'.format(self.current_date))
            # 5. Update dates.
            self.last_date = self.current_date
            self.current_date = self.next_date
            try:
                self.next_date = next(self.iter_dates)
            except StopIteration:
                self.logger.warning('Iter dates reached end, backtest over.')
                break

    def analyze(self):
        pass
