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

        # TODO - TEST
        self.alpha = 1.

    def handle_bar(self, codes, cur_date, cur_bar, cur_positions_weight, **kwargs):
        # Get margin stock.
        margin_stock = self.tar_position_scores.loc[cur_date].nlargest(self.margin)
        # Get keep stock.
        keep_stock_index = margin_stock.index.intersection(cur_positions_weight.index)
        # Get top k stock.
        top_k_stock = margin_stock.nlargest(self.top_k)
        # Get weights.
        weight = 1 / (len(keep_stock_index) + self.top_k)
        # Get target positions.
        tar_positions_weight = pd.Series(index=codes, data=0)
        tar_positions_weight[top_k_stock.index] = weight
        tar_positions_weight[margin_stock.index] = weight

        # TODO - TEST
        self.alpha += cur_bar['LABEL_0'].nlargest(self.top_k).mean() - cur_bar['LABEL_0'].mean()/ 2

        return tar_positions_weight






