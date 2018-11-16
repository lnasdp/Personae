# coding=utf-8

import os
import glob
import argparse
import numpy as np
import pandas as pd
from itertools import repeat

from personae.utility import factor
from personae.utility.profiler import TimeInspector

from concurrent.futures import ProcessPoolExecutor


# Load raw dfs.
def _load_raw_df(csv_path, _data_type='stock'):
    # Read raw csv.
    _df = pd.read_csv(csv_path, parse_dates=['date'], infer_datetime_format=True)
    # Rename code, for index data.
    _df = _df.rename(columns={'index_code': 'code'})
    # Upper case columns.
    _df.columns = map(str.upper, _df.columns)
    # Set date as index, and sort it.
    _df = _df.set_index('DATE').sort_index()
    # Calculate factors.
    for factor_name, calculator, args in factor.get_name_func_args_pairs(_data_type):
        _df[factor_name] = calculator(_df, *args)
    # Reset index.
    _df = _df.reset_index('DATE')
    return _df


def process_market_data(
        raw_data_dir,
        processed_data_dir,
        data_type='stock',
        market_type='all',
        market_cons_dir=None
):
    # Get csv dir.
    csv_dir = os.path.join(raw_data_dir, data_type)

    if market_type == 'all':
        csv_paths = glob.glob(os.path.join(csv_dir, '*.csv'))
    else:
        TimeInspector.set_time_mark()
        market_cons_path = os.path.join(market_cons_dir, '{}.pkl'.format(market_type))
        if not os.path.exists(market_cons_path):
            raise FileNotFoundError('{} not exist.'.format(market_cons_path))
        # Load market cons se.
        market_cons_se = pd.read_pickle(market_cons_path)  # type: pd.Series
        # Make csv paths.
        csv_paths = [os.path.join(csv_dir, '{}.csv'.format(code)) for code in market_cons_se.tolist()]
        TimeInspector.log_cost_time('Finished slicing {} cons.'.format(market_type))

    TimeInspector.set_time_mark()
    # dfs = [_load_raw_df(csv_paths[0], data_type)]
    with ProcessPoolExecutor(max_workers=16) as executor:
        dfs = list(executor.map(_load_raw_df, csv_paths, repeat(data_type)))
    TimeInspector.log_cost_time('Finished loading raw {} df, market type: {}'.format(data_type, market_type))

    # Concat and cache df.
    # TimeInspector.set_time_mark()
    # merged_data_path = os.path.join(merged_data_dir, '{}_{}.pkl'.format(market_type, data_type))
    df = pd.concat(dfs, sort=False)  # type: pd.DataFrame
    # df.to_pickle(merged_data_path)
    # TimeInspector.log_cost_time('Finished merging raw {} df to {}.'.format(data_type, merged_data_path))

    # Load merged df.
    # df = pd.read_pickle(merged_data_path)  # type: pd.DataFrame

    # Remove unused columns.
    columns = sorted(list(set(df.columns) - {'REPORT_TYPE', 'REPORT_DATE', 'ADJUST_PRICE_F'}))
    columns.sort()
    df = df[columns]

    # Replace inf with nan.
    TimeInspector.set_time_mark()
    df = df.replace([-np.inf, np.inf], np.nan)
    TimeInspector.log_cost_time('Finished replacing inf with nan.')

    # Drop nan columns.
    TimeInspector.set_time_mark()
    df = df.dropna()
    TimeInspector.log_cost_time('Finished dropping nan columns.')

    df = df.set_index(['DATE', 'CODE'])
    df = df.sort_index(level=[0, 1])

    # Calculate factors.
    TimeInspector.set_time_mark()

    TimeInspector.log_cost_time('Finished calculating LABEL_1, ALPHA.')
    df['LABEL_1'] = df['LABEL_0'].groupby(level='DATE').apply(lambda x: (x - x.mean()) / x.std())
    TimeInspector.set_time_mark()
    # Due to bug for pickle in OSX, https://stackoverflow.com/questions/31468117/
    # df.to_pickle(processed_data_path)

    # Make date range.
    date_range = pd.date_range('1999-01-01', '2019-01-01', freq='AS')
    cur_date = date_range[0]
    # Save split pkl.
    for next_date in date_range[1:]:
        # Processed year dir.
        processed_year_dir = os.path.join(processed_data_dir, str(cur_date.year))
        if not os.path.exists(processed_year_dir):
            os.makedirs(processed_year_dir)
        # Processed data path.
        processed_data_path = os.path.join(processed_year_dir, '{}_{}.pkl'.format(market_type, data_type))
        df_split = df.loc(axis=0)[cur_date: next_date, :]  # type: pd.DataFrame
        df_split.to_pickle(processed_data_path)
        cur_date = next_date
    TimeInspector.log_cost_time('Finished saving processed split {} df to {}'.format(data_type, processed_data_dir))


args_parser = argparse.ArgumentParser(prog='data_preprocessor')

args_parser.add_argument(
    '-t',
    '--market_type',
    type=str,
    required=False,
    default='all',
    help='Indicate which market data to process.'
)

args_parser.add_argument(
    '-c',
    '--market_cons_dir',
    type=str,
    required=False,
    help='Indicate where to load market constituents, only need when market type is assigned.'
)

args_parser.add_argument(
    '-r',
    '--raw_data_dir',
    type=str,
    required=True,
    help='Indicate where to load raw data.'
)

args_parser.add_argument(
    '-p',
    '--processed_data_dir',
    type=str,
    required=True,
    help='Indicate where to save preprocessed data.'
)


def main(args):

    market_type = 'csi500'

    # market_cons_dir = r'D:\Users\v-shuyw\data\ycz\data\market_data\market\processed'

    # raw_data_dir = r'D:\Users\v-shuyw\data\ycz\data\market_data\data\raw'
    # processed_data_dir = r'D:\Users\v-shuyw\data\ycz\data\market_data\data\processed'

    market_cons_dir = "/Users/shuyu/Desktop/Affair/Data/predictor/market/processed"

    raw_data_dir = "/Users/shuyu/Desktop/Affair/Data/predictor/market_data/raw"
    processed_data_dir = "/Users/shuyu/Desktop/Affair/Data/predictor/market_data/processed"

    process_market_data(
        raw_data_dir,
        processed_data_dir,
        data_type='stock',
        market_type=market_type,
        market_cons_dir=market_cons_dir
    )

    process_market_data(
        raw_data_dir,
        processed_data_dir,
        data_type='index'
    )


if __name__ == '__main__':
    # args_parsed = args_parser.parse_args()
    main(None)



