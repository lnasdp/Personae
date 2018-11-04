# coding=utf-8

import pandas as pd

import argparse
import glob
import os

from concurrent.futures import ProcessPoolExecutor

from personae.utility import profiler
from personae.utility import factor

factor_calculator_args_pairs = factor.get_factor_calculator_args_pairs()


def load_raw_df(csv_path):
    # 1. Load raw csv.
    df = pd.read_csv(csv_path, parse_dates=['date'], infer_datetime_format=True)
    # 2. Calculate factors.
    for factor_name, calculator, arg in factor_calculator_args_pairs:
        df[factor_name] = calculator(df, *arg)
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='process_raw_predictor_data')
    parser.add_argument('-d', '--csv_dir', required=True, type=str, help='csv dir indicates where to load raw csv.')
    parser.add_argument('-t', '--target_dir', required=True, type=str, help='target dir indicates where to save pkl.')

    args = parser.parse_args()
    csv_dir = args.csv_dir
    target_dir = args.target_dir

    csv_paths = glob.glob(os.path.join(csv_dir, '*.csv'))

    results = []

    profiler.TimeInspector.set_time_mark()
    with ProcessPoolExecutor(max_workers=16) as ex:
        res_iter = ex.map(load_raw_df, csv_paths)
    profiler.TimeInspector.log_cost_time('Finished loading all stock csv.')

    profiler.TimeInspector.set_time_mark()
    raw_df = pd.concat(list(res_iter))  # type: pd.DataFrame
    profiler.TimeInspector.log_cost_time('Finished concat all csv.')

    profiler.TimeInspector.set_time_mark()
    raw_df = raw_df.set_index(['code', 'date'])
    raw_df = raw_df.sort_index(axis=0, level=[0, 1])
    profiler.TimeInspector.log_cost_time('Finished setting index as code and date.')

    profiler.TimeInspector.set_time_mark()
    target_path = os.path.join(target_dir, 'all_market.pkl')
    raw_df.to_pickle(os.path.join(target_path))
    profiler.TimeInspector.log_cost_time('Finished dumping pkl to target path: {}'.format(target_path))

