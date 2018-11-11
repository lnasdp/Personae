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

    def __init__(self, data_dir, **kwargs):
        super(PredictorDataLoader, self).__init__(**kwargs)
        # 2. data dir
        self.data_dir = data_dir

        # 4. Dates.
        self.start_date = kwargs.get('start_date', '2005-01-01')
        self.end_date = kwargs.get('end_date', '2018-11-01')

    def load_data(self, codes='all', data_type='stock'):
        TimeInspector.set_time_mark()
        # 1. Load pkl.
        df = []
        date_range = pd.date_range(self.start_date, self.end_date, freq='AS')
        for cur_date in date_range:
            # Processed year dir.
            processed_year_dir = os.path.join(self.data_dir, str(cur_date.year))
            # Processed data path.
            processed_data_path = os.path.join(processed_year_dir, '{}.pkl'.format(data_type))
            df.append(pd.read_pickle(processed_data_path))
        df = pd.concat(df)
        # 2. Slice.
        if codes == 'all':
            df = df.loc(axis=0)[self.start_date: self.end_date, :]
        else:
            df = df.loc(axis=0)[self.start_date: self.end_date, codes]
        TimeInspector.log_cost_time('Finished loading data df.')
        # 3. Return.
        return df

