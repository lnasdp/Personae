# coding=utf-8

import multiprocessing as mp
import tushare as ts
import pandas as pd
import os

import config

from utility.logger import get_logger, TimeInspector
from utility.args_parser import spider_args_parser

spider_logger = get_logger('Spider')


class StockSpider(object):
    def __init__(self, instrument, start_date="2008-01-01", end_date="2018-08-01"):
        self.instrument = instrument
        self.start_date = start_date
        self.end_date = end_date

    def crawl(self):
        TimeInspector.set_time_mark()
        # Get stock frame.
        stock_frame = ts.get_k_data(code=self.instrument,
                                    start=self.start_date,
                                    end=self.end_date, retry_count=20)  # type: pd.DataFrame
        # Save to disk.
        if not os.path.exists(config.STOCK_DATA_DIR):
            os.makedirs(config.STOCK_DATA_DIR)
        stock_frame.to_csv(os.path.join(config.STOCK_DATA_DIR, '{}.csv'.format(self.instrument)))
        time = TimeInspector.get_cost_time()
        spider_logger.warning('Time cost: {0:.3f} | '
                              'Finish crawling instrument: {1}, from {2} to {3}'.
                              format(time,
                                     self.instrument,
                                     self.start_date,
                                     self.end_date)
                              )


if __name__ == '__main__':
    args = spider_args_parser.parse_args()
    # args.
    instruments = args.instruments
    start_date = args.start_date
    end_date = args.end_date
    # params.
    process_count = len(instruments) if len(instruments) < 8 else 8
    # crawling.
    pool = mp.Pool(processes=process_count)
    for instrument in instruments:
        spider = StockSpider(instrument, start_date='2018-01-01', end_date='2018-01-10')
        pool.apply_async(spider.crawl())
    pool.close()
    pool.join()

