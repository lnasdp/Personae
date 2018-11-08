# coding=utf-8

import numpy as np
import pandas as pd

from abc import abstractmethod


class BaseStrategy(object):

    @abstractmethod
    def before_trading(self, **kwargs):
        pass

    @abstractmethod
    def handle_bar(self, **kwargs):
        pass

    @abstractmethod
    def after_trading(self, **kwargs):
        pass


class RandomStrategy(BaseStrategy):

    def before_trading(self, **kwargs):
        pass

    def handle_bar(self, bar: pd.DataFrame):
        amount = [1000 if np.random.randint(0, 10) % 2 == 0 else -1000] * len(bar.index)
        return pd.Series(index=bar.index, data=amount)

    def after_trading(self, **kwargs):
        pass


class SimpleReturnStrategy(BaseStrategy):

    def before_trading(self, **kwargs):
        pass

    def handle_bar(self, bar: pd.DataFrame):
        top_stock = bar.nlargest(1, columns=['RETURN_SHIFT_0'])
        positions = pd.Series(index=bar.index, data=0)
        positions.loc[top_stock.index] = 100
        return positions

    def after_trading(self, **kwargs):
        pass


class TopKStrategy(BaseStrategy):

    def before_trading(self, **kwargs):
        pass

    def handle_bar(self, bar: pd.DataFrame):
        top_stock = bar.nlargest(1, columns=['RETURN_SHIFT_0'])
        positions = pd.Series(index=bar.index, data=0)
        positions.loc[top_stock.index] = 100
        return positions

    def after_trading(self, **kwargs):
        pass
