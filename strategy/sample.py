# coding=utf-8

import pandas as pd
import config

from base.strategy import BaseStrategy


class SampleStrategy(BaseStrategy):

    def __init__(self):
        super(SampleStrategy, self).__init__()
        self.times = 0

    def handle_bar(self, bar_df, date):
        self.times += 1
        bar_positions_series = pd.Series(index=bar_df.index, data=100 if self.times % 2 == 0 else -100)
        return bar_positions_series


if __name__ == '__main__':
    from utility.data_source import TuShareDataSource
    from utility.backtest import Backtest
    ds = TuShareDataSource()
    bt = Backtest(ds, SampleStrategy(), '2018-01-01', '2018-03-01')
    bt.start()
