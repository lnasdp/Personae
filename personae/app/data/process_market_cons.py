# coding=utf-8

import pandas as pd
import argparse
import os


def _rename_code(code_str):
    if code_str.startswith('6'):
        code_str = 'sh' + code_str
    else:
        code_str = 'sz' + code_str
    return code_str


def process_raw_df(raw_csv_path, processed_csv_path):
    # Read raw df.
    df = pd.read_csv(raw_csv_path, dtype=str)
    # Get code se.
    se = df.iloc(axis=1)[4]
    # Update code.
    se = se.apply(_rename_code)
    # Save market.
    se.to_pickle(processed_csv_path)


args_parser = argparse.ArgumentParser(
    prog='cons_preprocessor',
    description='Indicate market type, market cons dir, and raw, processed dirs to pre-process data.'
)

args_parser.add_argument(
    '-t',
    '--market_type',
    type=str,
    required=True,
    default='all',
    help='Indicate which market data to process.'
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
    # raw_csv_path = r'D:\Users\v-shuyw\data\ycz\data_sample\market\raw\csi500.csv'
    # processed_csv_path = r'D:\Users\v-shuyw\data\ycz\data_sample\market\processed\csi500.pkl'

    raw_csv_path = "/Users/shuyu/Desktop/Affair/Data/predictor/market/raw/csi500.csv"
    processed_csv_path = "/Users/shuyu/Desktop/Affair/Data/predictor/market/processed/csi500.pkl"

    process_raw_df(raw_csv_path, processed_csv_path)


if __name__ == '__main__':
    # args = args_parser.parse_args()
    main(None)
