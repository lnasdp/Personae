# coding=utf-8

import pandas as pd
import numpy as np

from abc import abstractmethod

from personae.contrib.model.model import BaseModel
from personae.contrib.data.handler import BaseDataHandler

from personae.utility.profiler import TimeInspector
from personae.utility.logger import get_logger


class BaseTrainer(object):

    def __init__(self, model_class, model_params: dict, data_handler: BaseDataHandler):

        # Model.
        self.model_class = model_class
        self.model_params = model_params
        self.model = None

        # Data handler.
        self.data_handler = data_handler

        # Update params about x, y space.
        self.x_space = self.data_handler.x_train.shape[1]
        self.y_space = self.data_handler.y_train.shape[1] if len(self.data_handler.y_train.shape) > 1 else 1

        self.model_params.update({
            "x_space": self.x_space,
            "y_space": self.y_space
        })

        # Logger.
        self.logger = get_logger('TRAINER')

    @abstractmethod
    def train(self):
        raise NotImplementedError('Implement this method to train models.')

    @abstractmethod
    def load(self):
        raise NotImplementedError('Implement this method to load models.')

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError('Implement this method to get prediction')


class StaticTrainer(BaseTrainer):

    def train(self):
        TimeInspector.set_time_mark()
        # Get model.
        model = self.model_class(**self.model_params)  # type: BaseModel
        model.fit(
            x_train=self.data_handler.x_train.values,
            y_train=self.data_handler.y_train.values.reshape((-1, self.y_space)),
            x_validation=self.data_handler.x_validation.values,
            y_validation=self.data_handler.y_validation.values.reshape((-1, self.y_space)),
            w_train=self.data_handler.w_train,
            w_validation=self.data_handler.w_validation
        )
        # Save model.
        model.save()
        # Set model.
        self.model = model
        TimeInspector.log_cost_time('Finished training model. (Static)')

    def load(self):
        TimeInspector.set_time_mark()
        # Get model.
        model = self.model_class(**self.model_params)  # type: BaseModel
        model.load()
        # Set model.
        self.model = model
        TimeInspector.log_cost_time('Finished loading model. (Static)')

    def predict(self):
        # Get predict scores.
        y_predict = self.model.predict(x_test=self.data_handler.x_test).reshape((-1, ))
        # Get test label.
        y_label = self.data_handler.y_test
        # Calculate ic.
        info = 'Finished get predict, ic: \n{}'
        self.logger.warning(info.format(
            np.corrcoef(y_predict, y_label)
        ))
        # TODO - Fix back.
        predict_scores = pd.Series(index=self.data_handler.x_test.index, data=y_label.values)
        return predict_scores


class RollingTrainer(BaseTrainer):

    def __init__(self, model_class, model_params: dict, data_handler: BaseDataHandler):
        super(RollingTrainer, self).__init__(model_class, model_params, data_handler)
        self.date_model_map = dict()

    def train(self):
        # Get total data parts.
        total_data_parts = self.data_handler.rolling_total_parts
        info = 'Total numbers of model are: {}, start training models...'
        self.logger.warning(info.format(total_data_parts))
        # Rolling train.
        for index, _ in enumerate(self.data_handler.rolling_iterator):
            TimeInspector.set_time_mark()
            # Get model.
            model = self.model_class(**self.model_params)  # type: BaseModel
            model.fit(
                x_train=self.data_handler.x_train.values,
                y_train=self.data_handler.y_train.values.reshape((-1, self.y_space)),
                x_validation=self.data_handler.x_validation.values,
                y_validation=self.data_handler.y_validation.values.reshape((-1, self.y_space)),
                w_train=self.data_handler.w_train,
                w_validation=self.data_handler.w_validation
            )
            # Save model.
            model.name = '{}_{}'.format(model.name, index)
            model.save()
            # Build date - model map.
            self.date_model_map[self.data_handler.rolling_test_end_dates[index]] = model
            info = 'Total numbers of model are: {},' ' finished training model: {}.'
            TimeInspector.log_cost_time(info.format(total_data_parts, index + 1))
        # Reset rolling data.
        self.data_handler.setup_rolling_data()

    def load(self):
        # Get total data parts.
        total_data_parts = self.data_handler.rolling_total_parts
        # Load models.
        for index in range(total_data_parts):
            model = self.model_class(**self.model_params)
            model.name = '{}_{}'.format(model.name, index)
            model.load()
            # Build date - model map.
            self.date_model_map[self.data_handler.rolling_test_end_dates[index]] = model

    def predict(self):
        predict_scores = []
        for index, _ in enumerate(self.data_handler.rolling_iterator):
            # Get model.
            model = self.date_model_map[self.data_handler.rolling_test_end_dates[index]]
            # Get predict score.
            _predict_scores = model.predict(x_test=self.data_handler.x_test)
            _predict_scores = pd.Series(index=self.data_handler.x_test.index, data=predict_scores)
            # Add predict score to scores.
            predict_scores.append(_predict_scores)
        # Concat result.
        predict_scores = pd.concat(predict_scores)
        return predict_scores

