# coding=utf-8

import tensorflow as tf
import lightgbm as gbm
import numpy as np
import os

from abc import abstractmethod


class BaseModel(object):

    def __init__(self, **kwargs):
        # Model name.
        self.name = kwargs.get('name', 'model')

        # Model save dir.
        self.save_dir = kwargs.get('save_dir', '/tmp/')

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass


class BaseNNModel(BaseModel):

    def __init__(self, **kwargs):
        super(BaseNNModel, self).__init__(**kwargs)

        # Session.
        self.session = kwargs.get('session', tf.Session())

        self.x_train = None
        self.y_train = None

        self.loss_func = None
        self.train_op = None

        # Seq length.
        self.seq_length = kwargs.get('seq_length', 5)

        # Batch size.
        self.batch_size = kwargs.get('batch_size', 64)

        # Save step.
        self.save_step = kwargs.get('save_step', 500)

        # Train steps.
        self.train_steps = kwargs.get('train_steps', 3000)

        # Dropout prob.
        self.dropout_prob = kwargs.get('dropout_prob', 0.6)

        # Learning rate.
        self.learning_rate = kwargs.get('learning_rate', 0.003)

    def fit(self, x_train, y_train, x_validate, y_validate):

        for train_step in range(self.train_steps):

            # 1. Run train op.
            loss, _ = self.session.run([self.loss_func, self.train_op], feed_dict={
                self.x_train: x_train,
                self.y_train: y_train
            })

            # 2. Log if need.
            pass


class LightGBM(BaseModel):

    def __init__(self, loss_func='mse', **kwargs):
        super(LightGBM, self).__init__(**kwargs)
        # Model.
        self.model = None

        # Boost round.
        self.boost_round = kwargs.get('boost_round', 1000)

        # Early stop round.
        self.early_stop_round = kwargs.get('early_stop_round', 50)

        # Booster parameters.
        self.booster_parameters = kwargs
        self.booster_parameters.update({'objective': loss_func})

    def fit(self,
            x_train,
            y_train,
            x_validation,
            y_validation,
            w_train=None,
            w_validation=None):

        # 1. Prepare train set.
        train_set = gbm.Dataset(x_train, label=y_train, weight=w_train)

        # 2. Prepare validation set.
        validation_set = gbm.Dataset(x_validation, label=y_validation, weight=w_validation)

        # 3. Prepare parameters.
        parameters = self.booster_parameters.copy()

        # 4. Evaluation result.
        eval_result = dict()

        # 5. Train.
        self.model = gbm.train(params=parameters,
                               train_set=train_set,
                               valid_sets=[validation_set],
                               evals_result=eval_result,
                               num_boost_round=self.boost_round,
                               early_stopping_rounds=self.early_stop_round,)

    def predict(self, x_test, **kwargs):
        self.assert_model()
        result = self.model.predict(x_test)
        return result

    def save(self, **kwargs):
        self.assert_model()
        self.model.save_model(filename=os.path.join(self.save_dir, self.name))

    def load(self, **kwargs):
        self.model = gbm.Booster(model_file=os.path.join(self.save_dir, self.name))

    def assert_model(self):
        if not self.model:
            raise ValueError('Model has not been trained, call `fit` to train model first.')
