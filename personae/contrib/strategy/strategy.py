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

    def __init__(self, tar_position_scores: pd.Series, top_k=100, **kwargs):
        super(MLTopKEqualWeightStrategy, self).__init__(**kwargs)
        # Target positions se, shift for trading execution.
        self.tar_position_scores = tar_position_scores.groupby(level=1).shift(1).fillna(0)
        # Top k.
        self.top_k = top_k

    def handle_bar(self, codes, cur_date, **kwargs):
        # Get top k.
        top_k_stock = self.tar_position_scores.loc[cur_date].nlargest(self.top_k)
        # Get target positions.
        tar_positions_weight = pd.Series(index=codes, data=0)
        tar_positions_weight[top_k_stock.index] = 1 / self.top_k
        return tar_positions_weight






