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


class HoldStrategy(BaseStrategy):

    def handle_bar(self, codes, **kwargs):
        amount = [10] * len(codes)
        return pd.Series(index=codes, data=amount)


class MLTopKStrategy(BaseStrategy):

    def __init__(self, tar_positions_se: pd.Series, top_k=100, **kwargs):
        super(MLTopKStrategy, self).__init__(**kwargs)
        # Target positions se, shift for trading execution.
        self.tar_positions_se = tar_positions_se.groupby(level=1).shift(-1).fillna(0)
        # Top k.
        self.top_k = top_k

    def handle_bar(self, codes, cur_date, **kwargs):
        # Get top k.
        top_k_stock = self.tar_positions_se.loc[cur_date].nlargest(self.top_k)
        # Get target positions.
        tar_positions = pd.Series(index=codes, data=0)
        tar_positions[top_k_stock.index] = 300
        return tar_positions



