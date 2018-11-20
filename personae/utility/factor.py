# coding=utf-8

import pandas as pd
import numpy as np


def returns(df: pd.DataFrame, key, shift_window=0):
    return df[key].pct_change().shift(shift_window)


def ewm(df: pd.DataFrame, key, alpha=0.9, span=5):

    def _ewm(x):
        w = alpha ** np.arange(0, len(x)).astype(np.float32)
        w /= w.sum()
        return np.nansum(w * x)

    return df[key].rolling(
        window=span,
        min_periods=1,
    ).apply(
        lambda x: _ewm(x),
        raw=True
    ).shift(
        -1 * span + 1
    )


def normalize(df: pd.DataFrame, tar_key, use_key):
    return df[tar_key] / df[use_key]


def diff(df: pd.DataFrame, key, window, normalized=False):
    se = df[key].diff(periods=window)
    return se if not normalized else se / df[key]


def rolling_std(df: pd.DataFrame, key, window, normalized=False):
    se = df[key].rolling(window=window).std()
    return se if not normalized else se / df[key]


def rolling_mean(df: pd.DataFrame, key, window, normalized=False):
    se = df[key].rolling(window=window).mean()
    return se if not normalized else se / df[key]


def rolling_max(df: pd.DataFrame, key, window, normalized=False):
    se = df[key].rolling(window=window).max()
    return se if not normalized else se / df[key]


def rolling_min(df: pd.DataFrame, key, window, normalized=False):
    se = df[key].rolling(window=window).min()
    return se if not normalized else se / df[key]


def rolling_quantile(df: pd.DataFrame, key, window, quantile, normalized=False):
    se = df[key].rolling(window=window).quantile(quantile)
    return se if not normalized else se / df[key]


def rolling_ic(df: pd.DataFrame, key_a, key_b, window):
    left_series, right_series = df[key_a], df[key_b]
    return left_series.rolling(window=window).corr(right_series)


def get_name_func_args_pairs(data_type='stock'):

    close = 'ADJUST_PRICE' if data_type == 'stock' else 'CLOSE'

    name_func_args_pairs, windows = [], [3, 5, 10, 15, 20, 25, 30, 60]

    # return.
    name_func_args_pairs.append(('RETURN', returns, [close, 0]))
    name_func_args_pairs.append(('LABEL_RETURN', returns, [close, -2]))
    name_func_args_pairs.append(('LABEL_EWM_RETURN', ewm, ['LABEL_RETURN']))

    for w in windows:
        factor_name = '{}_VOLUME_IC_{}'.format(close, w)
        name_func_args_pairs.append((factor_name, rolling_ic, [close, 'VOLUME', w]))

    if data_type == 'stock':
        fields = [
            'OPEN',
            'HIGH',
            'LOW',
            'ADJUST_PRICE',
            'VOLUME',
        ]
    else:
        fields = [
            'CLOSE',
            'VOLUME'
        ]

    for field in fields:
        # field_diff_window.
        for w in windows:
            factor_name = '{}_DIFF_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, diff, [field, w, True]))

        # field_rolling_std_window.
        for w in windows:
            factor_name = '{}_STD_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_std, [field, w, True]))

        # field_rolling_mean_window.
        for w in windows:
            factor_name = '{}_MEAN_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_mean, [field, w, True]))

        # field_rolling_mean_window.
        for w in windows:
            factor_name = '{}_MAX_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_max, [field, w, True]))

        # field_rolling_min_window.
        for w in windows:
            factor_name = '{}_MIN_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_min, [field, w, True]))

        # field_rolling_quantile_num_window.
        for w in windows:
            factor_name = '{}_QUANTILE_25_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_quantile, [field, w, 0.25, True]))

    name_func_args_pairs.append(('OPEN', normalize, ['OPEN', 'CLOSE']))
    name_func_args_pairs.append(('HIGH', normalize, ['HIGH', 'CLOSE']))
    name_func_args_pairs.append(('LOW', normalize, ['LOW', 'CLOSE']))

    return name_func_args_pairs

