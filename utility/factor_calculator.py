# coding=utf-8


def calculate_return_rate(origin_df, bar_gap=-1):
    return_rate = origin_df['close'] / origin_df.groupby(level=0)['close'].shift(bar_gap) - 1
    return return_rate


def calculate_price_diff(origin_df, bar_gap=1):
    price_diff = origin_df['close'] - origin_df.groupby(level=0)['close'].shift(bar_gap)
    return price_diff
