# coding=utf-8

import argparse
import pandas as pd


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


if __name__ == '__main__':
    raw_csv_path = r'D:\Users\v-shuyw\data\ycz\data_sample\market\raw\csi500\000905cons.csv'
    processed_csv_path = r'D:\Users\v-shuyw\data\ycz\data_sample\market\processed\csi500\csi500.pkl'
    process_raw_df(raw_csv_path, processed_csv_path)

