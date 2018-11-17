# coding=utf-8

import logging

from personae.contrib.backtest.engine import BaseEngine
from personae.contrib.data.loader import PredictorDataLoader
from personae.contrib.strategy.strategy import EqualWeightHoldStrategy


if __name__ == '__main__':

    stock_df = PredictorDataLoader.load_data(
        data_dir=r'D:\Users\v-shuyw\data\ycz\data\market_data\data\processed',
        market_type='csi500',
        data_type='stock'
    )

    bench_df = PredictorDataLoader.load_data(
        data_dir=r'D:\Users\v-shuyw\data\ycz\data\market_data\data\processed',
        market_type='all',
        data_type='index'
    )

    e = BaseEngine(
        stock_df=stock_df,
        bench_df=bench_df,
        start_date='2015-07-01',
        end_date='2017-07-01',
        cash=1000000,
        sh_level=logging.INFO)

    e.run(EqualWeightHoldStrategy())
    e.analyze()
    e.plot()
