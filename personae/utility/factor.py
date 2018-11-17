# coding=utf-8

import pandas as pd


def returns(df: pd.DataFrame, key, shift_window=0):
    return df[key].pct_change().shift(shift_window)


def ewm(df: pd.DataFrame, key, span=5):
    return df[key].ewm(span=span).mean()


def diff(df: pd.DataFrame, key, window):
    return df[key].diff(periods=window)


def rolling_std(df: pd.DataFrame, key, window):
    return df[key].rolling(window=window).std()


def rolling_mean(df: pd.DataFrame, key, window):
    return df[key].rolling(window=window).mean()


def rolling_max(df: pd.DataFrame, key, window):
    return df[key].rolling(window=window).max()


def rolling_min(df: pd.DataFrame, key, window):
    return df[key].rolling(window=window).min()


def rolling_quantile(df: pd.DataFrame, key, window, quantile):
    return df[key].rolling(window=window).quantile(quantile) / df[key]


def rolling_ic(df: pd.DataFrame, key_a, key_b, window):
    left_series, right_series = df[key_a], df[key_b]
    return left_series.rolling(window=window).corr(right_series)


def get_name_func_args_pairs(data_type='stock'):

    close = 'ADJUST_PRICE' if data_type == 'stock' else 'CLOSE'

    name_func_args_pairs, windows = [], [3, 5, 10, 20, 30, 60]

    # 1. return.
    name_func_args_pairs.append(('RETURN', returns, [close, 0]))
    name_func_args_pairs.append(('LABEL_RETURN', returns, [close, -2]))
    # name_func_args_pairs.append(('LABEL_EWM_RETURN', ewm, [close, 5]))

    # 2. adjust_price_volume_rolling_ic_window.
    for w in windows:
        factor_name = '{}_VOLUME_IC_{}'.format(close, w)
        name_func_args_pairs.append((factor_name, rolling_ic, [close, 'VOLUME', w]))

    if data_type == 'stock':
        fields = [
            'ADJUST_PRICE',
            'VOLUME',
            'CHANGE',
            'TURNOVER',
            'MONEY',
            'TRADED_MARKET_VALUE',
            'PE_TTM',
            'PS_TTM',
            'PC_TTM',
            'PB'
        ]
    else:
        fields = [
            'CLOSE',
            'VOLUME'
        ]

    for field in fields:
        # 3. field_diff_window.
        for w in windows:
            factor_name = '{}_DIFF_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, diff, [field, w]))

        # 4. field_rolling_std_window.
        for w in windows:
            factor_name = '{}_STD_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_std, [field, w]))

        # 5. field_rolling_mean_window.
        for w in windows:
            factor_name = '{}_MEAN_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_mean, [field, w]))

        # 6. field_rolling_max_window.
        for w in windows:
            factor_name = '{}_MAX_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_max, [field, w]))

        # 7. field_rolling_min_window.
        for w in windows:
            factor_name = '{}_MIN_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_min, [field, w]))

        # 8. field_rolling_quantile_num_window.
        for w in windows:
            factor_name = '{}_QUANTILE_25_{}'.format(field, w)
            name_func_args_pairs.append((factor_name, rolling_quantile, [field, w, 0.25]))

    return name_func_args_pairs

