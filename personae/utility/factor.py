# coding=utf-8

import pandas as pd


def returns(df: pd.DataFrame, key, shift_window=0):
    return df.shift(periods=shift_window)[key] / df.shift(periods=shift_window+1)[key] - 1


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


name_func_args_pairs, windows = [], [5, 10, 30, 60, 120]

# 1. return.
name_func_args_pairs.append(('RETURN_SHIFT_0', returns, ['CLOSE', 0]))
name_func_args_pairs.append(('LABEL_0', returns, ['CLOSE', -2]))

# 2. close_volume_rolling_ic_window.
for w in windows:
    factor_name = 'CLOSE_VOLUME_IC_{}'.format(w)
    name_func_args_pairs.append((factor_name, rolling_ic, ['CLOSE', 'VOLUME', w]))

for field in ['CLOSE', 'VOLUME']:
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

