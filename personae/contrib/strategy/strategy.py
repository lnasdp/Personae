# coding=utf-8

import numpy as np
import pandas as pd

from personae.contrib.model.model import BaseModel
from abc import abstractmethod


class BaseStrategy(object):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def handle_bar(self, **kwargs):
        pass


class EqualWeightHoldStrategy(BaseStrategy):

    def handle_bar(self, codes, **kwargs):
        return pd.Series(index=codes, data=1 / len(codes))


class MLTopKEqualWeightStrategy(BaseStrategy):

    def __init__(self, tar_position_scores: pd.Series, top_k=50, margin=300, **kwargs):
        super(MLTopKEqualWeightStrategy, self).__init__(**kwargs)
        # Target positions se, shift for trading execution.
        self.tar_position_scores = tar_position_scores.groupby(level=1).shift(1).fillna(0)
        # Top k.
        self.top_k = top_k
        # Margin.
        self.margin = margin

    def handle_bar(self, codes, cur_date, cur_positions_weight, **kwargs):
        # Get prediction.
        scores = self.tar_position_scores.loc[cur_date]
        # Get margin stocks.
        margin_stocks = scores.nlargest(self.margin)
        # Get current stocks.
        cur_stocks = cur_positions_weight[cur_positions_weight > 0]
        # Get current stocks index in and out margin.
        cur_stocks_in_margin_index = margin_stocks.index.isin(cur_stocks)
        cur_stocks_not_in_margin_index = ~cur_stocks_in_margin_index
        # Get top k stocks.
        top_k_stocks = margin_stocks.nlargest(self.top_k)
        # Get hold stocks.
        hold_stocks = margin_stocks[cur_stocks_in_margin_index]
        # Get buy stocks.
        buy_stocks = margin_stocks[cur_stocks_not_in_margin_index]

        # Get weights.
        weight = 1 / (len(top_k_stocks) + self.top_k)
        # Get target positions.
        tar_positions_weight = pd.Series(index=codes, data=0)
        tar_positions_weight[top_k_stocks.index] = weight
        return tar_positions_weight






