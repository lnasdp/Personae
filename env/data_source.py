# coding=utf-8

import pandas as pd

from base.data_source import BaseDataSource


class TestDataSource(BaseDataSource):

    def _load_instruments(self):
        self.instruments = ['600001', '600002']

    def _load_origin_data(self):
        columns = ['instrument', 'datetime', 'open', 'close', 'high', 'low', 'volume', 'label']
        data = [
            ('600001', '2010-01-01', 3.5, 4.0, 4.1, 3.6, 10000, 1),
            ('600001', '2010-01-02', 4.0, 4.4, 4.6, 3.9, 9000, 0),
            ('600001', '2010-01-03', 4.4, 4.1, 4.5, 3.8, 8000, 1),
            ('600002', '2010-01-02', 4.4, 4.1, 4.5, 3.8, 8000, 0),
            ('600002', '2010-01-04', 4.1, 4.4, 4.6, 3.9, 9000, 0),
            ('600002', '2010-01-05', 3.4, 4.0, 4.1, 3.6, 10000, 1),
            ('600003', '2010-01-04', 4.0, 4.4, 4.6, 3.9, 8000, 1),
            ('600003', '2010-01-02', 4.4, 4.3, 4.5, 3.8, 9000, 0),
            ('600003', '2010-01-05', 4.3, 3.0, 4.1, 3.0, 10000, 1),
        ]

        self.origin_df = pd.DataFrame(data=data, columns=columns)
        self.origin_df['datetime'] = pd.to_datetime(self.origin_df['datetime'])
        self.origin_df = self.origin_df.set_index(['instrument', 'datetime'])
        self.origin_df = self.origin_df.loc(axis=0)[self.instruments]


class TuShareDataSource(BaseDataSource):
    pass


class RiceQuantDataSource(BaseDataSource):
    pass


if __name__ == '__main__':
    ds = TestDataSource()
