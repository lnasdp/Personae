# coding=utf-8


def calculate_alpha(origin_df, bar_gap=-1):
    alpha = origin_df['close'] - origin_df.groupby(level=0)['close'].shift(bar_gap)
    return alpha
