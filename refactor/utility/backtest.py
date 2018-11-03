# coding=utf-8

import pandas as pd
import numpy as np
import logging
import config
import os

from utility.logger import get_logger


class Backtest(object):

    def __init__(self,
                 data_source,
                 strategy,
                 start_date,
                 end_date,
                 cash=1.e8,
                 slippage=0.05,
                 bench_mark='000905',
                 trade_fee_rate=0.004,
                 sh_level=logging.INFO):

        # 1. Date related property.
        self.start_date = start_date
        self.end_date = end_date

        # 2. Data source and data_handler related property.
        self.data_source = data_source
        self.features_df = self.data_source.features_df.loc(axis=0)[:, self.start_date: self.end_date]
        self.benchmark = bench_mark

        # 3. Iteration related property.
        self.available_dates = self.features_df.index.get_level_values(level=1).unique().tolist()
        self.iter_dates = iter(self.available_dates)

        # 4. Strategy.
        self.strategy = strategy

        # 5. Market related property.
        self.cash = cash
        self.initial_cash = cash
        self.slippage = slippage
        self.trade_fee_rate = trade_fee_rate
        self.positions = pd.Series(index=self.features_df.index, data=0)

        # 6. Metrics.
        self.metric_df = pd.DataFrame(index=self.features_df.index.get_level_values(level=1).unique()[:-1],
                                      columns=[config.ROE,
                                               config.RETURN_RATE,
                                               config.ALPHA,
                                               config.BETA,
                                               config.BETA_ROE,
                                               config.CASH,
                                               config.PROFITS,
                                               config.HOLDINGS],
                                      data=0,
                                      dtype=float)
        # 7. Others.
        self.logger = get_logger('Backtest', sh_level=sh_level)

    def start(self):

        # 1. Check available dates.
        if len(self.available_dates) < 2:
            raise ValueError('Invalid iter dates length, less than 2 bars.')

        # 2. Init iter dates.
        last_date, current_date, next_date = pd.Timestamp('1970-01-01'), next(self.iter_dates), next(self.iter_dates)

        # 3. Init market info and setting.
        cash = self.cash
        initial_cash = cash
        last_roe = 1.0
        last_beta_roe = 1.0
        last_bar_positions = self.positions.loc(axis=0)[:, current_date]
        last_bar_price = self.features_df.loc(axis=0)[:, current_date][config.CLOSE]

        try:
            last_benchmark_bar_price = self.features_df.loc(axis=0)[self.benchmark, current_date][config.CLOSE]
        except KeyError:
            raise KeyError('Benchmark: {}, not found, please run spider to crawl it first.'.format(self.benchmark))

        # 4. Start backtest.
        while True:

            # 4.1. Apply before_trading()
            self.logger.debug('On date: {}.'.format(current_date))
            if current_date.day > last_date.day:
                self.strategy.before_trading(current_date)
                self.logger.debug('`before_trading` called.')

            # 4.2 Apply handle_bar() and get bar positions.
            current_bar_df = self.features_df.loc(axis=0)[:, current_date]  # type: pd.DataFrame
            current_bar_positions = self.strategy.handle_bar(current_bar_df, current_date)  # type: pd.Series
            current_benchmark_bar_price = current_bar_df.loc(axis=0)[self.benchmark, current_date][config.CLOSE]
            self.logger.debug('`handle_bar` called.')

            # 4.3. Update positions.
            self.positions.loc(axis=0)[:, current_date] = current_bar_positions

            # 4.4. Ger current bar price.
            current_bar_price = current_bar_df[config.CLOSE]

            # 4.5. Calculate last and current positions diff.
            positions_diff = current_bar_positions.values - last_bar_positions.values

            # 4.6. Calculate trade cost.
            trade_cost = (current_bar_price * positions_diff).sum()

            # 4.7. Calculate trade fee.
            trade_fee = trade_cost * self.trade_fee_rate + positions_diff.sum() * self.slippage

            # 4.8. Calculate last and current bar price diff.
            bar_price_diff = current_bar_price.values - last_bar_price.values

            # 4.9. Calculate bar profits.
            profits = (bar_price_diff * last_bar_positions).sum()

            # 4.10. Update cash.
            cash -= (trade_cost + trade_fee)
            cash += profits

            # 4.11. Update holding values.
            holdings = (current_bar_price * current_bar_positions).sum()

            # 4.12. Calculate roe.
            roe = (profits + cash + holdings) / initial_cash

            # 4.13. Calculate return rate.
            return_rate = roe - last_roe

            # 4.14. Calculate beta.
            beta = current_benchmark_bar_price / last_benchmark_bar_price - 1

            # 4.13. Calculate alpha.
            alpha = return_rate - beta

            # 4.15. Calculate beta roe.
            beta_roe = last_beta_roe * (1 + beta)

            self.logger.info('RoE: {0:.4f} | '
                             'Return Rate: {1:.4f} | '
                             'Cash: {2:.2f} | '
                             'Holdings: {3:.2f} | '
                             'Profits: {4:.2f} | '
                             'Fee: {5:.2f} | '
                             'Cost: {6:.2f}'.format(roe, return_rate, cash, holdings, profits, trade_fee, trade_cost))

            # 4.13 Update metric.
            metric_values = roe, return_rate, alpha, beta, beta_roe, cash, profits, holdings
            self.metric_df.loc(axis=0)[current_date][config.ROE,
                                                     config.RETURN_RATE,
                                                     config.ALPHA,
                                                     config.BETA,
                                                     config.BETA_ROE,
                                                     config.CASH,
                                                     config.PROFITS,
                                                     config.HOLDINGS] = metric_values

            # 5. Apply after_trading().
            if current_date.day + 1 == next_date.day:
                self.strategy.after_trading(current_date)
                self.logger.debug('`after_trading` called.')

            # 6. Update dates.
            last_roe = roe
            last_beta_roe = beta_roe
            last_date = current_date
            current_date = next_date
            last_bar_price = current_bar_price
            last_bar_positions = current_bar_positions
            last_benchmark_bar_price = current_benchmark_bar_price

            try:
                next_date = next(self.iter_dates)
            except StopIteration:
                self.logger.warning('Iter dates reached end, backtest over.')
                break

    def analyze(self, verbose=True):
        # Return rate series.
        return_rate = self.metric_df[config.RETURN_RATE]
        # RoE.
        roe = self.metric_df[config.ROE][-1]
        # Beta roe.
        beta_roe = self.metric_df[config.BETA_ROE][-1]
        # Sharpe.
        sharpe = return_rate.mean() / return_rate.std() * np.sqrt(250)
        # Beta.
        beta = self.metric_df[config.BETA]
        # Beta sharpe.
        beta_sharpe = beta.mean() / beta.std() * np.sqrt(250)
        # AnR
        annualized_return = roe / (len(self.available_dates) - 1) / 250
        # Beta AnR
        annualized_beta = beta_roe / (len(self.available_dates) - 1) / 250
        # Max Drawdown.
        mdd = ((return_rate.cumsum() - return_rate.cumsum().cummax()) / (1 + return_rate.cumsum().cummax())).min()
        # Beta max drawdown.
        beta_mdd = ((beta.cumsum() - beta.cumsum().cummax()) / (1 + beta.cumsum().cummax())).min()
        # Result.
        result_metric = pd.DataFrame(index=pd.MultiIndex.from_product([['Strategy', 'Beta'], ['RoE', 'Sharpe', 'AnR', 'MDD']]),
                                     data=[roe,
                                           sharpe,
                                           annualized_return,
                                           mdd,
                                           beta_roe,
                                           beta_sharpe,
                                           annualized_beta,
                                           beta_mdd],
                                     columns=['Indicator'])
        if verbose:
            self.logger.warning('Performance metric:\n{}'.format(result_metric))
        return result_metric
