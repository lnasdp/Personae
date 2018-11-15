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


def _load_raw_df(csv_path, data_type='stock'):
    # Read raw csv.
    df = pd.read_csv(csv_path, parse_dates=['date'], infer_datetime_format=True)
    # Rename code, for index data.
    df = df.rename(columns={'index_code': 'code'})
    # Upper case columns.
    df.columns = map(str.upper, df.columns)
    # Set date as index, and sort it.
    df = df.set_index('DATE').sort_index()
    # Calculate factors.
    for factor_name, calculator, args in factor.get_name_func_args_pairs(data_type):
        df[factor_name] = calculator(df, *args)
    # Reset index.
    df = df.reset_index('DATE')
    return df


def merge_raw_df(raw_data_dir, merged_data_dir, data_type='stock'):
    # Get csv dir.
    csv_dir = os.path.join(raw_data_dir, data_type)

    # Get all csv paths.
    csv_paths = glob.glob(os.path.join(csv_dir, '*.csv'))

    # 3. Load raw dfs.
    TimeInspector.set_time_mark()
    with ProcessPoolExecutor(max_workers=16) as executor:
        dfs = list(executor.map(_load_raw_df, csv_paths, repeat(data_type)))
    TimeInspector.log_cost_time('Finished loading raw {} df.'.format(data_type))

    # 4. Concat and cache df.
    TimeInspector.set_time_mark()
    merged_data_path = os.path.join(merged_data_dir, '{}.pkl'.format(data_type))
    df = pd.concat(dfs, sort=False)  # type: pd.DataFrame
    df.to_pickle(merged_data_path)
    TimeInspector.log_cost_time('Finished merging raw {} df to {}.'.format(data_type, merged_data_path))


def process_merged_df(cache_data_dir, processed_dir, data_type='stock'):
    # Load merged df.
    df = pd.read_pickle(os.path.join(cache_data_dir, '{}.pkl'.format(data_type)))  # type: pd.DataFrame

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
    df['LABEL_1'] = df['LABEL_0'].groupby(level='CODE').apply(lambda x: (x - x.mean()) / x.std())
    TimeInspector.set_time_mark()
    # Due to bug for pickle in OSX, https://stackoverflow.com/questions/31468117/
    # df.to_pickle(processed_data_path)

    # Make date range.
    date_range = pd.date_range('1999-01-01', '2019-01-01', freq='AS')
    cur_date = date_range[0]
    # Save split pkl.
    for next_date in date_range[1:]:
        # Processed year dir.
        processed_year_dir = os.path.join(processed_dir, str(cur_date.year))
        if not os.path.exists(processed_year_dir):
            os.makedirs(processed_year_dir)
        # Processed data path.
        processed_data_path = os.path.join(processed_year_dir, '{}.pkl'.format(data_type))
        df_split = df.loc(axis=0)[cur_date: next_date, :]  # type: pd.DataFrame
        df_split.to_pickle(processed_data_path)
        cur_date = next_date
    TimeInspector.log_cost_time('Finished saving processed split {} df to {}'.format(data_type, processed_dir))


args_parser = argparse.ArgumentParser(prog='data_preprocessor')
args_parser.add_argument('-r',
                         '--raw_data_dir',
                         type=str,
                         required=True,
                         help='Indicate where to load raw data.')

args_parser.add_argument('-p',
                         '--processed_data_dir',
                         type=str,
                         required=True,
                         help='Indicate where to save preprocessed data.')

args_parser.add_argument('-m',
                         '--merged_data_dir',
                         type=str,
                         required=True,
                         help='Indicate where to save merged data.')


if __name__ == '__main__':

    # args_parsed = args_parser.parse_args()
    #
    # raw_data_dir = args_parsed.raw_data_dir
    # merged_data_dir = args_parsed.merged_data_dir
    # processed_data_dir = args_parsed.processed_data_dir

    # raw_data_dir = r'D:\Users\v-shuyw\data\ycz\data\raw'
    # merged_data_dir = r'D:\Users\v-shuyw\data\ycz\data\merged'
    # processed_data_dir = r'D:\Users\v-shuyw\data\ycz\data\processed'

    raw_data_dir = r'D:\Users\v-shuyw\data\ycz\data_sample\raw'
    merged_data_dir = r'D:\Users\v-shuyw\data\ycz\data_sample\merged'
    processed_data_dir = r'D:\Users\v-shuyw\data\ycz\data_sample\processed'

    # raw_data_dir = '/Users/shuyu/Desktop/Affair/Temp/data/raw'
    # merged_data_dir = '/Users/shuyu/Desktop/Affair/Temp/data/merged'
    # processed_data_dir = '/Users/shuyu/Desktop/Affair/Temp/data/processed'

    # raw_data_dir = '/Users/shuyu/Desktop/Affair/Temp/data_tmp/raw'
    # merged_data_dir = '/Users/shuyu/Desktop/Affair/Temp/data_tmp/merged'
    # processed_data_dir = '/Users/shuyu/Desktop/Affair/Temp/data_tmp/processed'

    merge_raw_df(raw_data_dir, merged_data_dir, data_type='stock')
    merge_raw_df(raw_data_dir, merged_data_dir, data_type='index')

    process_merged_df(merged_data_dir, processed_data_dir, data_type='stock')
    process_merged_df(merged_data_dir, processed_data_dir, data_type='index')
