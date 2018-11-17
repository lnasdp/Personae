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

    def handle_bar(
            self,
            codes,
            cur_date,
            cur_close,
            cur_total_assets,
            cur_positions_weight,
            cur_positions_amount,
            **kwargs
    ):
        # Get prediction.
        scores = self.tar_position_scores.loc[cur_date]
        # Get margin scores.
        margin_scores = scores.nlargest(self.margin)
        # Get current valid position weights.
        valid_positions_weight = cur_positions_weight[cur_positions_weight > 0]
        # Get current valid position weights in margin.
        valid_margin_positions_weight = margin_scores[valid_positions_weight.index].dropna()
        # Assign margin weights.
        tar_positions_weight = pd.Series(index=codes, data=0)
        tar_positions_weight = self._update_margin_positions_weight(
            tar_positions_weight,
            cur_positions_amount,
            cur_close,
            cur_total_assets,
            valid_margin_positions_weight,
        )
        # Set new buy weight.
        weight_last = 1 - np.sum(tar_positions_weight)
        # Get top k stock.
        top_k_scores = margin_scores.nlargest(self.top_k)
        # Get top k stock not in current valid in margin positions.
        invalid_top_k_scores = valid_margin_positions_weight[top_k_scores.index].dropna()
        # Get different set.
        valid_top_k_scores_index = list(set(top_k_scores.index) - set(invalid_top_k_scores.index))
        # Get new top k.
        valid_top_k_scores = top_k_scores.loc(axis=0)[valid_top_k_scores_index]
        # Update target positions.
        if len(valid_top_k_scores) > 0 and np.sum(scores) != 0:
            tar_positions_weight[valid_top_k_scores.index] = weight_last / len(valid_top_k_scores)
        return tar_positions_weight

    @staticmethod
    def _update_margin_positions_weight(
            tar_positions_weight,
            cur_positions_amount,
            cur_close,
            cur_total_asset,
            valid_margin_positions_weight,
    ):
        # Get valid margin positions amount.
        valid_margin_positions_amount = cur_positions_amount[valid_margin_positions_weight.index].dropna()
        # w2 = c2 * p2 / t2
        c2_mul_p2_over_t2 = (valid_margin_positions_amount * cur_close).dropna() / cur_total_asset
        tar_positions_weight[c2_mul_p2_over_t2.index] = c2_mul_p2_over_t2
        return tar_positions_weight




