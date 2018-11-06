# coding=utf-8

import os
import glob
import argparse
import pandas as pd

from personae.utility import factor
from personae.utility.profiler import TimeInspector

from concurrent.futures import ProcessPoolExecutor


def _load_raw_df(csv_path):
    df = pd.read_csv(csv_path)
    return df


def concat_raw_df(raw_data_dir, cache_data_dir, data_type='stock'):

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
    cache_data_path = os.path.join(cache_data_dir, '{}.pkl'.format(data_type))
    df = pd.concat(dfs)  # type: pd.DataFrame
    df.to_pickle(cache_data_path)
    TimeInspector.log_cost_time('Finished saving {} df to {}.'.format(data_type, cache_data_path))


def process_raw_df(cache_data_dir, data_type='stock'):
    # 1. Load raw df.
    df = pd.read_pickle(os.path.join(cache_data_dir, '{}.pkl'.format(data_type)))  # type: pd.DataFrame

    # 2. Remove unused columns.
    columns = list(set(df.columns) - {'report_type', 'report_date', 'adjust_price_f'})
    columns.sort()
    df = df[columns]

    # 2. Calculate factors.
    TimeInspector.set_time_mark()
    for factor_name, calculator, args in factor.name_calculator_args_pairs:
        df[factor_name] = calculator(df, *args)
    TimeInspector.log_cost_time('Finished calculating factors')

    df = df.set_index(['code', 'date'])
    df = df.sort_index(level=[0, 1])

    print(df)


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

args_parser.add_argument('-c',
                         '--cache_data_dir',
                         type=str,
                         required=True,
                         help='Indicate where to save cache data.')


if __name__ == '__main__':

    args_parsed = args_parser.parse_args()

    concat_raw_df(args_parsed.raw_data_dir, args_parsed.cache_data_dir, data_type='stock')
    concat_raw_df(args_parsed.raw_data_dir, args_parsed.cache_data_dir, data_type='index')

    process_raw_df(args_parsed.cache_data_dir, data_type='stock')
