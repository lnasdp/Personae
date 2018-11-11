# coding=utf-8

import numpy as np
import pandas as pd

from personae.contrib.model.model import BaseModel
from abc import abstractmethod


class BaseStrategy(object):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def before_trading(self, **kwargs):
        pass

    @abstractmethod
    def handle_bar(self, **kwargs):
        pass

    @abstractmethod
    def after_trading(self, **kwargs):
        pass


class HoldStrategy(BaseStrategy):

    def before_trading(self, **kwargs):
        pass

    def handle_bar(self, bar: pd.DataFrame, cur_date, **kwargs):
        amount = [100 ] * len(bar.index)
        return pd.Series(index=bar.index, data=amount)

    def after_trading(self, **kwargs):
        pass


class RandomStrategy(BaseStrategy):

    def before_trading(self, **kwargs):
        pass

    def handle_bar(self, bar: pd.DataFrame, cur_date, **kwargs):
        amount = [1000 if np.random.randint(0, 10) % 2 == 0 else -1000] * len(bar.index)
        return pd.Series(index=bar.index, data=amount)

    def after_trading(self, **kwargs):
        pass


class SimpleReturnStrategy(BaseStrategy):

    def before_trading(self, **kwargs):
        pass

    def handle_bar(self, bar: pd.DataFrame, cur_date, **kwargs):
        top_stock = bar.nlargest(1, columns=['RETURN_SHIFT_0'])
        positions = pd.Series(index=bar.index, data=0)
        positions.loc[top_stock.index] = 100
        return positions

    def after_trading(self, **kwargs):
        pass


class TopKStrategy(BaseStrategy):

    def __init__(self, top_k=10, **kwargs):
        super(TopKStrategy, self).__init__(**kwargs)
        self.top_k = top_k

    def before_trading(self, **kwargs):
        pass

    def handle_bar(self, bar: pd.DataFrame, cur_date, **kwargs):
        top_stock = bar.nlargest(self.top_k, columns=['RETURN_SHIFT_0'])
        positions = pd.Series(index=bar.index, data=0)
        positions.loc[top_stock.index] = 100
        return positions

    def after_trading(self, **kwargs):
        pass


class MLTopKStrategy(BaseStrategy):

    def __init__(self, predict_se: pd.Series, top_k=10, **kwargs):
        super(MLTopKStrategy, self).__init__(**kwargs)

        # Predict.
        self.predict_se = predict_se

        # Top k.
        self.top_k = top_k

    def before_trading(self, **kwargs):
        pass

    def handle_bar(self, bar: pd.DataFrame, cur_date, **kwargs):
        # Get top k.
        top_stock = self.predict_se.loc[cur_date:, ].nlargest(self.top_k)
        # Get target positions.
        positions = pd.Series(index=bar.index, data=0)
        positions.loc[top_stock.index] = 100

        return positions

    def after_trading(self, **kwargs):
        pass



