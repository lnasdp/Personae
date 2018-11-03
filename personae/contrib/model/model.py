# coding=utf-8

import tensorflow as tf

from abc import abstractmethod

from personae.utility.log import TimeInspector


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