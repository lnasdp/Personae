# coding=utf-8

import pandas as pd


class Backtest(object):

    def __init__(self, data_source, strategy, start_date, end_date, cash=1e8, slippage=0.4, trade_fee_rate=0.004):
        # 1. Data source.
        self.data_source = data_source
        # 2. Strategy.
        self.strategy = strategy
        # 3. Dates.
        self.start_date = start_date
        self.end_date = end_date
        # 4. Market setting.
        self.cash = cash
        self.slippage = slippage
        self.trade_fee_rate = trade_fee_rate
        # 5. Positions.
        self.positions = pd.DataFrame(index=self.data_source.origin_df.index, columns=['amount'], data=0)

    def start(self):
        pass

    def analyze(self):
        pass


if __name__ == '__main__':
    from utility.data_source import TuShareDataSource
    ds = TuShareDataSource()
    bt = Backtest(ds, None, '2018-01-01', '2018-03-01')
    bt.start()
