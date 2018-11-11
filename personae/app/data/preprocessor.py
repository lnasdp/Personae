# coding=utf-8

import os
import glob
import argparse
import numpy as np
import pandas as pd

from personae.utility import factor
from personae.utility.profiler import TimeInspector

from concurrent.futures import ProcessPoolExecutor


def _load_raw_df(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'], infer_datetime_format=True)
    df = df.rename(columns={'index_code': 'code'})
    df.columns = map(str.upper, df.columns)
    return df


def merge_raw_df(raw_data_dir, merged_data_dir, data_type='stock'):
    # 1. Get csv dir.
    csv_dir = os.path.join(raw_data_dir, data_type)

    # 2. Get all csv paths.
    csv_paths = glob.glob(os.path.join(csv_dir, '*.csv'))

    # 3. Load raw dfs.
    TimeInspector.set_time_mark()
    with ProcessPoolExecutor(max_workers=16) as executor:
        dfs = list(executor.map(_load_raw_df, csv_paths))
    TimeInspector.log_cost_time('Finished loading raw {} df.'.format(data_type))

    # 4. Concat and cache df.
    TimeInspector.set_time_mark()
    cache_data_path = os.path.join(merged_data_dir, '{}.pkl'.format(data_type))
    df = pd.concat(dfs)  # type: pd.DataFrame
    df.to_pickle(cache_data_path)
    TimeInspector.log_cost_time('Finished merging raw {} df to {}.'.format(data_type, cache_data_path))


def process_merged_df(cache_data_dir, processed_dir, data_type='stock'):
    # Load merged df.
    df = pd.read_pickle(os.path.join(cache_data_dir, '{}.pkl'.format(data_type)))  # type: pd.DataFrame

    # Remove unused columns.
    columns = list(set(df.columns) - {'REPORT_TYPE', 'REPORT_DATE', 'ADJUST_PRICE_F'})
    columns.sort()
    df = df[columns]

    # Calculate factors.
    TimeInspector.set_time_mark()
    for factor_name, calculator, args in factor.name_func_args_pairs:
        df[factor_name] = calculator(df, *args)
    TimeInspector.log_cost_time('Finished calculating factors')

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

    processed_data_path = os.path.join(processed_dir, '{}.pkl'.format(data_type))

    TimeInspector.set_time_mark()
    df.to_pickle(processed_data_path)
    TimeInspector.log_cost_time('Finished saving processed {} df to {}'.format(data_type, processed_data_path))


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

    merge_raw_df(raw_data_dir, merged_data_dir, data_type='stock')
    merge_raw_df(raw_data_dir, merged_data_dir, data_type='index')

    process_merged_df(merged_data_dir, processed_data_dir, data_type='stock')
    process_merged_df(merged_data_dir, processed_data_dir, data_type='index')
