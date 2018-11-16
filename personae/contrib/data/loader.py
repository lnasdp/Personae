# coding=utf-8

import pandas as pd
import os

from abc import abstractmethod

from personae.utility.profiler import TimeInspector


class BaseDataLoader(object):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def load_data(self, **kwargs):
        pass


class PredictorDataLoader(BaseDataLoader):

    @classmethod
    def load_data(
            cls,
            data_dir,
            market_type='all',
            data_type='stock',
            start_date='2005-01-01',
            end_date='2018-11-01'
    ):
        TimeInspector.set_time_mark()
        # 1. Load pkl.
        df = []
        date_range = pd.date_range(start_date, end_date, freq='AS')
        for cur_date in date_range:
            # Processed year dir.
            processed_year_dir = os.path.join(data_dir, str(cur_date.year))
            # Processed data path.
            processed_data_path = os.path.join(processed_year_dir, '{}_{}.pkl'.format(market_type, data_type))
            df.append(pd.read_pickle(processed_data_path))
        df = pd.concat(df)  # type: pd.DataFrame
        df = df.sort_index(level=['DATE', 'CODE'])
        # 2. Slice.
        df = df.loc(axis=0)[start_date: end_date, :]
        TimeInspector.log_cost_time('Finished loading df, market type: {}, data type: {}.'.format(
            market_type,
            data_type
        ))
        # 3. Return.
        return df

