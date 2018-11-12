# coding=utf-8

import tensorflow as tf
import lightgbm as gbm
import numpy as np
import shutil
import os

from tensorflow.contrib.layers import variance_scaling_initializer
from sklearn.metrics import roc_auc_score, mean_squared_error
from abc import abstractmethod

from personae.utility.logger import get_logger


class BaseModel(object):

    def __init__(self, **kwargs):
        # Model name.
        self.name = kwargs.get('name', 'model')

        # Model save dir.
        self.save_dir = kwargs.get('save_dir', '/tmp/')

        # Model save path.
        self.save_path = os.path.join(self.save_dir, self.name)

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass


class BaseNNModel(BaseModel):

    def __init__(self, x_space, y_space, **kwargs):
        super(BaseNNModel, self).__init__(**kwargs)

        # Session.
        self.session = kwargs.get('session', tf.Session())

        # Input shape.
        self.x_space = x_space
        self.y_space = y_space

        # Input tensor.
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.x_space], name='x_input')
        self.y_input = tf.placeholder(tf.float32, shape=[None, self.y_space], name='y_input')

        # Output tensor.
        self.y_predict = None

        # Loss func and train op.
        self.loss_func = None
        self.train_op = None

        # Model persistence.
        self.saver = None
        self.builder_signature = None

        # Seq length.
        self.seq_length = kwargs.get('seq_length', 5)
        # Batch size.
        self.batch_size = kwargs.get('batch_size', 64)
        # Save step.
        self.save_step = kwargs.get('save_step', 100)
        # Train steps.
        self.train_steps = kwargs.get('train_steps', 3000)
        # Dropout prob.
        self.dropout_prob = kwargs.get('dropout_prob', 0.6)
        # Learning rate.
        self.learning_rate = kwargs.get('learning_rate', 0.003)

        self.save_graph = kwargs.get('save_graph', False)

        # Logger.
        self.logger = get_logger('{}'.format(self.name).upper())

    def fit(self, x_train, y_train, x_validate, y_validate):
        # Init best validation loss.
        best_validation_loss = np.inf
        # Start iteration.
        for train_step in range(self.train_steps):
            # Get batch.
            data_size = len(x_train)
            # Get mini batch.
            indices = np.random.choice(data_size, size=self.batch_size)
            x_batch = x_train[indices]
            y_batch = y_train[indices]
            # Run train op.
            train_loss, _ = self.session.run([self.loss_func, self.train_op], feed_dict={
                self.x_input: x_batch,
                self.y_input: y_batch
            })
            # Save and early stop if need.
            if (train_step + 1) % self.save_step == 0:
                # Evaluate validation set.
                validation_loss = self.evaluate(x_validate, y_validate)
                info = 'Train step: {0} | Reach checkpoint, train loss: {1:.5f}, validation loss: {2:.5f}'
                self.logger.warning(info.format(
                    train_step + 1,
                    train_loss,
                    validation_loss)
                )
                # Update best validation loss if need.
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    info = 'Train step: {0} | Best validation loss updated: {1:.5f}, save model.'
                    self.logger.warning(info.format(
                        train_step,
                        best_validation_loss
                    ))
                    self.save()
                # Early stop if need.
                if validation_loss > best_validation_loss * 0.8:
                    info = 'Train step: {0} | current validation loss is worse than 0.8 of the best, early stop.'
                    self.logger.warning(info.format(
                        train_step
                    ))
                    self.load()
                    break

    def evaluate(self, x_input, y_input, **kwargs):
        loss = self.session.run(self.loss_func, feed_dict={
            self.x_input: x_input,
            self.y_input: y_input,
        })
        return loss

    def save(self, **kwargs):
        # Save checkpoint.
        self.saver.save(self.session, self.save_path)
        # Save graph if need.
        if self.save_graph and self.builder_signature:
            graph_dir = os.path.join(self.save_dir, 'graph')
            if os.path.exists(graph_dir):
                shutil.rmtree(graph_dir)
            builder = tf.saved_model.builder.SavedModelBuilder(graph_dir)
            builder_key_signature_map = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.builder_signature
            }
            builder.add_meta_graph_and_variables(
                sess=self.session,
                tag=[tf.saved_model.tag_constants.SERVING],
                signature_def_map=builder_key_signature_map
            )
            builder.save()

    def load(self, **kwargs):
        self.saver.restore(self.session, self.save_path)

    def predict(self, x_test, **kwargs):
        y_predict = self.session.run(self.y_predict, feed_dict={
            self.x_input: x_test
        })
        return y_predict


class MLPModel(BaseNNModel):

    def __init__(self,
                 num_hidden_layers=3,
                 num_hidden_units=None,
                 loss_func='mse',
                 **kwargs):
        super(MLPModel, self).__init__(**kwargs)

        # Hidden units.
        if num_hidden_units is None:
            num_hidden_units = [256, 256, 256]

        # Weight initializer.
        weight_initializer = variance_scaling_initializer()

        dense = None

        # Layers.
        for layer_index in range(num_hidden_layers):

            if layer_index == 0:
                inputs = self.x_input
            else:
                inputs = dense

            dense = tf.layers.dense(inputs=inputs,
                                    units=num_hidden_units[layer_index],
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=weight_initializer,
                                    name='dense_{}'.format(layer_index))
        # Predict tensor.
        self.y_predict = tf.layers.dense(inputs=dense,
                                         units=self.y_space,
                                         kernel_initializer=weight_initializer,
                                         name='y_predict')
        # Loss func.
        if loss_func == 'mse':
            self.loss_func = tf.losses.mean_squared_error(self.y_input, self.y_predict)
        else:
            raise NotImplementedError('Unsupported loss func name: {}.'.format(loss_func))

        # Optimizer and train op.
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss_func)

        self.session.run(tf.global_variables_initializer())

        # Builder.
        self.builder_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'x_input': tf.saved_model.utils.build_tensor_info(self.x_input),
                'y_input': tf.saved_model.utils.build_tensor_info(self.y_input)
            },
            outputs={
                'y_predict': tf.saved_model.utils.build_tensor_info(self.y_predict),
                'loss_func': tf.saved_model.utils.build_tensor_info(self.loss_func)
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        self.saver = tf.train.Saver()


class LightGBMModel(BaseModel):

    def __init__(self, loss_func='mse', **kwargs):
        super(LightGBMModel, self).__init__(**kwargs)
        # Model.
        self.model = None

        # Loss func.
        self.loss_func = loss_func

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
                               verbose_eval=50,
                               valid_sets=[train_set, validation_set],
                               evals_result=eval_result,
                               num_boost_round=self.boost_round,
                               early_stopping_rounds=self.early_stop_round,)

    def predict(self, x_test, **kwargs):
        self.assert_model()
        y_predict = self.model.predict(x_test)
        return y_predict

    def evaluate(self, x_test, y_test, **kwargs):
        self.assert_model()
        y_predict = self.predict(x_test)
        if self.loss_func == 'mse':
            return mean_squared_error(y_test, y_predict)
        else:
            return roc_auc_score(y_test, y_predict)

    def save(self, **kwargs):
        self.assert_model()
        self.model.save_model(filename=self.save_path)

    def load(self, **kwargs):
        self.model = gbm.Booster(model_file=self.save_path)

    def assert_model(self):
        if not self.model:
            raise ValueError('Model has not been trained, call `fit` to train model first.')
