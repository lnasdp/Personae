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


class MLTopKMarginEqualWeightStrategy(BaseStrategy):

    def __init__(self, tar_position_scores: pd.Series, top_k=50, margin=300, **kwargs):
        super(MLTopKMarginEqualWeightStrategy, self).__init__(**kwargs)
        # Target positions se, shift for trading execution.
        self.tar_position_scores = tar_position_scores.groupby(level=1).shift(1).fillna(0)
        # Top k.
        self.top_k = top_k
        # Margin.
        self.margin = margin

    def handle_bar(self, codes, cur_date, cur_positions_weight, **kwargs):
        # Get prediction.
        scores = self.tar_position_scores.loc[cur_date]
        # Get margin scores.
        margin_scores = scores.nlargest(self.margin)
        # Get current valid position weights.
        valid_positions_weight = cur_positions_weight[cur_positions_weight > 0]
        # Get current valid position weights in margin.
        valid_positions_weight_in_margin = margin_scores.loc(axis=0)[
            valid_positions_weight.index.intersection(margin_scores.index)
        ]
        # Assign margin weights.
        tar_positions_weight = pd.Series(index=codes, data=0)
        """
        Get intersection of index, for codes in two days might not be the same.
        """
        index_common = tar_positions_weight.index.intersection(valid_positions_weight_in_margin.index)
        # Set margin weight.
        tar_positions_weight[index_common] = valid_positions_weight_in_margin[index_common]
        # Set new buy weight.
        weight_last = 1 - np.sum(tar_positions_weight)
        # Get top k stock.
        top_k_scores = margin_scores.nlargest(self.top_k)
        # Get top k stock not in current valid in margin positions.
        invalid_top_k_scores = valid_positions_weight_in_margin.loc(axis=0)[
            top_k_scores.index.intersection(valid_positions_weight_in_margin.index)
        ]
        # Get different set.
        valid_top_k_scores_index = list(set(top_k_scores.index) - set(invalid_top_k_scores.index))
        # Get new top k.
        valid_top_k_scores = top_k_scores.loc(axis=0)[valid_top_k_scores_index]
        # Update target positions.
        if len(valid_top_k_scores) > 0 and np.sum(scores) != 0:
            tar_positions_weight[valid_top_k_scores.index] = weight_last / len(valid_top_k_scores)
        return tar_positions_weight






