# coding=utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from personae.utility import logger
from personae.utility.profiler import TimeInspector


class BaseEngine(object):

    def __init__(self,
                 stock_df,
                 bench_df,
                 start_date='2005-01-01',
                 end_date='2018-11-01',
                 cash=1.e8,
                 prefer=0.95,
                 charge=0.0015,
                 slippage=0.01,
                 benchmark='sh000905',
                 **kwargs):

        # Data df.
        self.stock_df = stock_df.loc(axis=0)[start_date: end_date, :]
        self.bench_df = bench_df.loc(axis=0)[start_date: end_date, benchmark]

        # Strategy.
        self.strategy = None

        # Backtest range.
        self.start_date = start_date
        self.end_date = end_date

        # Backtest codes.
        self.available_codes = self.stock_df.index.get_level_values(level=1).unique().tolist()
        self.available_dates = self.bench_df.index.get_level_values(level=0).unique().tolist()

        if len(self.available_codes) < 1:
            raise ValueError('Available codes less than 1, please check data.')

        if len(self.available_dates) < 2:
            raise ValueError('Available dates less than 3, please check data.')

        # Backtest dates.
        self.iter_dates = iter(self.available_dates)

        # Backtest trade info.
        self.cash = cash
        self.prefer = prefer
        self.charge = charge
        self.slippage = slippage

        # Logger.
        self.logger = logger.get_logger('BACKTEST', **kwargs)

    def run(self, strategy):

        # Strategy.
        self.strategy = strategy

        # Iter dates.
        self.iter_dates = iter(self.available_dates)

        # Cash.
        cash, initial_cash = self.cash, self.cash

        # Cur date.
        cur_date = next(self.iter_dates)

        # Last close
        last_close = self.stock_df.loc(axis=0)[cur_date]['ADJUST_PRICE']

        # Positions.
        positions_weight_dic = dict()
        positions_amount_dic = dict()

        positions_weight_dic[cur_date] = pd.Series(
            index=last_close.index.get_level_values(level='CODE'),
            data=0
        )

        positions_amount_dic[cur_date] = pd.Series(
            index=last_close.index.get_level_values(level='CODE'),
            data=0
        )

        # Performance.
        roe_dic = dict()
        return_dic = dict()
        total_assets_dic = dict()

        total_assets_dic[cur_date] = cash
        return_dic[cur_date] = 0.
        roe_dic[cur_date] = 1.

        bench_returns_se = self.bench_df['CLOSE'].reset_index('CODE', drop=True).pct_change().fillna(0)
        bench_roe_se = (bench_returns_se + 1).cumprod()

        # Last date.
        last_date, cur_date = cur_date, self._get_next_date(self.iter_dates)

        self.logger.warning('Start backtesting...')

        TimeInspector.set_time_mark()
        while cur_date:

            # Get cur bar.
            try:
                cur_bar = self.stock_df.loc(axis=0)[cur_date]  # type: pd.Series
                # Get current available codes.
                cur_codes = cur_bar.index.get_level_values(level='CODE')
                # Get cur bar return, close.
                cur_close = cur_bar['ADJUST_PRICE']
            except KeyError:
                # Update positions with last.
                positions_weight_dic[cur_date] = positions_weight_dic[last_date]
                positions_amount_dic[cur_date] = positions_amount_dic[last_date]
                # Update performance with last.
                roe_dic[cur_date] = roe_dic[last_date]
                return_dic[cur_date] = 0.
                total_assets_dic[cur_date] = total_assets_dic[last_date]
                # Update last date and current date.
                last_date, cur_date = cur_date, self._get_next_date(self.iter_dates)
                # Log and continue.
                self.logger.info('All stock cannot trade on: {}, continue.'.format(cur_date))
                continue

            # Get current positions weight and amount (last date).
            cur_positions_weight = positions_weight_dic[last_date]
            cur_positions_amount = positions_amount_dic[last_date]

            # Forward fill close.
            cur_close = self._forward_fill_cur_close(cur_close, last_close)

            # Get profit.
            profit = self._calculate_profit(
                cur_positions_amount,
                cur_close,
                last_close
            )

            # Get holdings value.
            holdings_value = self._calculate_holdings_value(cur_positions_amount, cur_close)

            # Get total assets.
            total_assets = cash + holdings_value

            # Get current available codes.
            cur_available_codes = cur_codes[~(cur_bar['RETURN'].abs() > 0.98)]
            cur_available_close = cur_close[cur_available_codes]

            # Call handle bar.
            tar_positions_weight = self.strategy.handle_bar(**{
                'codes': cur_available_codes,
                'cur_date': cur_date,
                'cur_close': cur_available_close,
                'cur_prefer': self.prefer,
                'cur_total_assets': total_assets,
                'cur_positions_weight': cur_positions_weight,
                'cur_positions_amount': cur_positions_amount,
            })

            # Forward fill target positions weight.
            tar_positions_weight = self._zero_fill_tar_positions_weight(tar_positions_weight, cur_positions_weight)
            tar_positions_amount = self._calculate_positions_amount(
                tar_positions_weight,
                cur_close,
                total_assets,
                self.prefer,
            )

            # Check target positions weight.
            self.check_tar_positions_weight(tar_positions_weight)

            positions_weight_dic[cur_date] = tar_positions_weight
            positions_amount_dic[cur_date] = tar_positions_amount

            # Get loss.
            loss = self._calculate_loss(
                tar_positions_amount,
                cur_positions_amount,
                cur_close,
                self.slippage,
                self.charge
            )

            # Update total assets.
            total_assets -= loss
            # total_assets -= 0

            # Update holdings value.
            holdings_value = self._calculate_holdings_value(tar_positions_amount, cur_close)

            # Update cash.
            cash = total_assets - holdings_value

            # Update roe.
            roe = total_assets / initial_cash

            # Update roe.
            roe_dic[cur_date] = roe

            # Update return.
            returns = roe - roe_dic[last_date]
            return_dic[cur_date] = returns

            # Update total assets.
            total_assets_dic[cur_date] = total_assets

            log_info = 'Date: {0} | ' \
                       'Profit: {1:.3f} | ' \
                       'Loss: {2:.3f} | ' \
                       'Cash: {3:.3f} | ' \
                       'Holdings: {4:.3f} | ' \
                       'RoE: {5:.3f} | ' \
                       'Returns: {6:.6f}'

            self.logger.warning(log_info.format(
                cur_date,
                profit,
                loss,
                cash,
                holdings_value,
                roe,
                returns)
            )

            last_date, cur_date, last_close = cur_date, self._get_next_date(self.iter_dates), cur_close
        TimeInspector.log_cost_time('Finished backtest.')
        return pd.Series(roe_dic), pd.Series(return_dic), bench_roe_se, bench_returns_se

    def analyze(self, long_roe_se, long_returns_se, bench_roe_se, bench_returns_se):
        # Inner func.
        def _analyze(returns_se, roe_se):
            returns_mean = returns_se.mean()
            returns_std = returns_se.std()
            annual = returns_mean * 250
            sharpe = returns_mean / returns_std * np.sqrt(250)
            mdd = ((returns_se.cumsum() - returns_se.cumsum().cummax()) / (1 + returns_se.cumsum().cummax())).min()
            performance = pd.Series({
                'annual': annual,
                'sharpe': sharpe,
                'mdd': mdd,
                'roe': roe_se.iloc[-1],
            })
            return performance
        # Long.
        long_performance = _analyze(long_returns_se, long_roe_se)
        # Bench.
        bench_performance = _analyze(bench_returns_se, bench_roe_se)
        # Hedge.
        hedge_returns_se = long_returns_se - bench_returns_se
        hedge_roe_se = ((long_roe_se - bench_roe_se) + 1)
        hedge_performance = _analyze(hedge_returns_se, hedge_roe_se)
        # Result.
        result = {
            'Long': long_performance,
            'Bench': bench_performance,
            'Hedge': hedge_performance,
        }
        # Log.
        self.logger.warning('\n{}\n'.format(
            pd.DataFrame(result)
        ))
        # Plot.
        plt.figure()
        roe_concat_df = pd.concat([
            hedge_roe_se,
            long_roe_se,
            bench_roe_se
        ], axis=1)
        roe_concat_df.columns = ['HEDGE', 'LONG', 'BENCH']
        roe_concat_df.plot()
        plt.show()

    @staticmethod
    def check_tar_positions_weight(tar_positions):
        if tar_positions.sum() > 1.001 or tar_positions.sum() < -1.001:
            raise ValueError('Invalid target positions weight: {}, please check your strategy.'.format(
                tar_positions.sum()
            ))

    @staticmethod
    def _calculate_positions_amount(positions_weight, close, total_assets, prefer):
        # Trade cash.
        trade_cash = prefer * total_assets
        # Positions cash.
        positions_cash = trade_cash * positions_weight
        # Positions amount.
        positions_amount = positions_cash.div(close, fill_value=0)
        # Normalize amount.
        positions_amount = positions_amount - positions_amount % 100
        # positions_amount = ((positions_amount / 100) + 1e-12).apply(np.floor) * 100
        return positions_amount

    @staticmethod
    def _calculate_holdings_value(positions_amount, close):
        # Calculate holdings value.
        holdings_value = np.sum(positions_amount.mul(close, fill_value=0))
        return holdings_value

    @staticmethod
    def _calculate_profit(positions: pd.Series, cur_close: pd.Series, last_close: pd.Series):
        # Close diff.
        close_diff = cur_close.sub(last_close, fill_value=0)
        # Profit.
        profit = np.sum(positions.mul(close_diff, fill_value=0))
        return profit

    @staticmethod
    def _calculate_loss(tar_positions_amount, cur_positions_amount, cur_close, slippage, charge):
        # Calculate positions amount diff.
        positions_amount_diff = tar_positions_amount.sub(cur_positions_amount, fill_value=0)
        # Calculate loss.
        loss_slippage = np.sum(positions_amount_diff.abs() * slippage)
        loss_charge = np.sum(positions_amount_diff.abs() * cur_close * charge)
        loss = loss_slippage + loss_charge
        return loss

    @staticmethod
    def _forward_fill_cur_close(cur_close, last_close):
        close_concat = pd.concat(
            [last_close, cur_close],
            axis=1,
            sort=False
        ).fillna(
            method='ffill',
            axis=1
        )
        close_concat.columns = ['LAST', 'CUR']
        cur_close = close_concat.loc(axis=1)['CUR']
        return cur_close

    @staticmethod
    def _zero_fill_tar_positions_weight(tar_positions_weight, cur_positions_weight):
        positions_weight_concat = pd.concat(
            [cur_positions_weight, tar_positions_weight],
            axis=1,
            sort=False
        ).fillna(
            value=0,
            axis=1
        )
        positions_weight_concat.columns = ['CUR', 'TAR']
        tar_positions_weight = positions_weight_concat.loc(axis=1)['TAR']
        return tar_positions_weight

    @staticmethod
    def _get_next_date(iter_dates):
        try:
            next_date = next(iter_dates)
        except StopIteration:
            next_date = None
        return next_date

