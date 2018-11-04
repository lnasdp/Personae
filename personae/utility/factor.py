# coding=utf-8

import pandas as pd


def calculate_return(df: pd.DataFrame, key):
    return df[key] / df.shift(periods=1)[key] - 1


def calculate_diff(df: pd.DataFrame, key, window):
    return df[key].diff(periods=window) / df[key]


def calculate_rolling_std(df: pd.DataFrame, key, window):
    return df[key].rolling(window=window).std() / df[key]


def calculate_rolling_mean(df: pd.DataFrame, key, window):
    return df[key].rolling(window=window).mean() / df[key]


def calculate_rolling_max(df: pd.DataFrame, key, window):
    return df[key].rolling(window=window).max() / df[key]


def calculate_rolling_min(df: pd.DataFrame, key, window):
    return df[key].rolling(window=window).min() / df[key]


def calculate_rolling_rank(df: pd.DataFrame, key, window):
    return df[key].rolling(window=window).rank()


def calculate_rolling_quantile(df: pd.DataFrame, key, window, quantile):
    return df[key].rolling(window=window).quantile(quantile)


def calculate_rolling_ic(df: pd.DataFrame, key_a, key_b, window):
    left_series, right_series = df[key_a], df[key_b]
    return left_series.rolling(window=window).corr(right_series)


def get_factor_calculator_args_paris():
    factor_calculator_args_pairs, windows = [], [5, 10, 30, 60, 120]

    # 1. return.
    factor_calculator_args_pairs.append(('return', calculate_return, ['close']))

    # 2. close_volume_rolling_ic_window.
    for window in windows:
        factor_name = 'close_rolling_ic_{}'.format(window)
        factor_calculator_args_pairs.append((factor_name, calculate_rolling_ic, ['close', 'volume', window]))

    for field in ['close', 'volume']:
        # 2. field_diff_window.
        for window in windows:
            factor_name = '{}_diff_{}'.format(field, window)
            factor_calculator_args_pairs.append((factor_name, calculate_diff, [field, window]))

        # 3. field_rolling_std_window.
        for window in windows:
            factor_name = '{}_rolling_std_{}'.format(field, window)
            factor_calculator_args_pairs.append((factor_name, calculate_rolling_std, [field, window]))

        # 4. field_rolling_mean_window.
        for window in windows:
            factor_name = '{}_rolling_mean_{}'.format(field, window)
            factor_calculator_args_pairs.append((factor_name, calculate_rolling_mean, [field, window]))

        # 5. field_rolling_max_window.
        for window in windows:
            factor_name = '{}_rolling_max_{}'.format(field, window)
            factor_calculator_args_pairs.append((factor_name, calculate_rolling_max, [field, window]))

        # 6. field_rolling_min_window.
        for window in windows:
            factor_name = '{}_rolling_min_{}'.format(field, window)
            factor_calculator_args_pairs.append((factor_name, calculate_rolling_min, [field, window]))

        # 7. field_rolling_rank_window.
        for window in windows:
            factor_name = '{}_rolling_rank_{}'.format(field, window)
            factor_calculator_args_pairs.append((factor_name, calculate_rolling_rank, [field, window]))

        # 8. field_rolling_quantile_num_window.
        for window in windows:
            factor_name = '{}_rolling_quantile_25_{}'.format(field, window)
            factor_calculator_args_pairs.append((factor_name, calculate_rolling_quantile, [field, window, 0.25]))

    return factor_calculator_args_pairs
