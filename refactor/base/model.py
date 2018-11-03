# coding=utf-8

import tensorflow as tf
import numpy as np
import shutil
import math
import os

import config

from utility import logger
from abc import abstractmethod


class BaseModel(object):

    def __init__(self, model_name, x_space, y_space, **options):
        # 1. Model related.
        self.model_name = model_name
        self.x_space = x_space
        self.y_space = y_space
        self.train_steps = 0
        # 1.1 Input.
        self.x_input = None
        self.y_input = None
        # 1.2. Output.
        self.y_predict = None
        # 1.3. Loss func and Optimizer.
        self.loss_func = None
        self.optimizer = None
        # 1.4. Parameters.
        self.t_dropout_keep_prob = None
        self.t_is_training = None
        # 2. Training related.
        self._init_options(options)
        # 3. Saving related.
        self.model_dir = os.path.join(config.SAVED_MODEL_DIR, self.model_name, config.DATETIME_LOAD_MODULE)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # 3.3 Graph builder.
        self.graph_dir = os.path.join(self.model_dir, 'graph')
        self.builder_signature = None
        # 4. Logger.
        self.logger = logger.get_logger(model_name)
        # 5. ABC method.
        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_signature()
        # 6. Saver.
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoint')
        self.checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        self.saver = tf.train.Saver()
        # 7. Summary Writer.
        self.summary_dir = os.path.join(self.model_dir, 'summaries')
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.summary_writer = tf.summary.FileWriter(self.summary_dir, graph=self.session.graph)
        self.summary_merge_all_op = tf.summary.merge_all()

    def _init_options(self, options):

        try:
            self.session_config = options[config.KEY_SESSION_CONFIG]
        except KeyError:
            # TODO - Add default.
            self.session_config = None

        try:
            self.session = options[config.KEY_SESSION]
        except KeyError:
            self.session = tf.Session(config=self.session_config)

        try:
            self.batch_size_train = options[config.KEY_BATCH_SIZE_TRAIN]
        except KeyError:
            self.batch_size_train = 256

        try:
            self.batch_size_evaluate = options[config.KEY_BATCH_SIZE_EVALUATE]
        except KeyError:
            self.batch_size_evaluate = self.batch_size_train * 100

        try:
            self.learning_rate = options[config.KEY_LEARNING_RATE]
        except KeyError:
            self.learning_rate = 0.003

        try:
            self.dropout_prob = options[config.KEY_DROPOUT_PROB]
        except KeyError:
            self.dropout_prob = 0.6

        try:
            self.train_steps_limit = options[config.KEY_TRAIN_STEPS_LIMIT]
        except KeyError:
            self.train_steps_limit = 2000

        try:
            self.train_save_steps = options[config.KEY_TRAIN_SAVE_STEPS]
        except KeyError:
            self.train_save_steps = 500

        try:
            self.loss_func_name = options[config.KEY_LOSS_FUNC_NAME]
        except KeyError:
            self.loss_func_name = 'MSE'

        try:
            self.seq_length = options[config.KEY_SEQ_LENGTH]
        except KeyError:
            self.seq_length = 5

        try:
            self.rolling_start_dates = options[config.KEY_ROLLING_START_DATES]
        except KeyError:
            self.rolling_start_dates = None

        try:
            self.rolling_end_dates = options[config.KEY_ROLLING_END_DATES]
        except KeyError:
            self.rolling_end_dates = None

    def save(self, need_save_graph=False):
        self.logger.warning("Saver reach checkpoint.")
        self.saver.save(self.session, self.checkpoint_path)
        if need_save_graph and self.builder_signature:
            if os.path.exists(self.graph_dir):
                shutil.rmtree(self.graph_dir)
            builder = tf.saved_model.builder.SavedModelBuilder(self.graph_dir)
            builder.add_meta_graph_and_variables(self.session,
                                                 [tf.saved_model.tag_constants.SERVING],
                                                 {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.builder_signature})
            builder.save()

    def restore(self):
        self.saver.restore(self.session, self.checkpoint_path)

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass

    @abstractmethod
    def _init_signature(self):
        pass

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def predict(self, s):
        pass


class BaseRLModel(BaseModel):

    def __init__(self, model_name, s_space, a_space, **options):
        super(BaseRLModel, self).__init__(model_name, s_space, a_space, **options)
        self.buffer_count = 0

    def _init_options(self, options):
        super(BaseRLModel, self)._init_options(options)

        try:
            self.train_episodes_limit = options[config.KEY_TRAIN_EPISODES_LIMIT]
        except KeyError:
            self.train_episodes_limit = 1000

        try:
            self.eval_episodes = options[config.KEY_EVAL_EPISODES]
        except KeyError:
            self.eval_episodes = 1

        try:
            self.save_episode = options[config.KEY_SAVE_EPISODES]
        except KeyError:
            self.save_episode = 50

        try:
            self.buffer_size = options[config.KEY_BUFFER_SIZE]
        except KeyError:
            self.buffer_size = 3000

        try:
            self.gamma = options[config.KEY_GAMMA]
        except KeyError:
            self.gamma = 0.95

        try:
            self.tau = options[config.KEY_TAU]
        except KeyError:
            self.tau = 0.01

        try:
            self.epsilon = options[config.KET_EPSILON]
        except KeyError:
            self.epsilon = 0.9

    @abstractmethod
    def snapshot(self, s, a, r, s_n):
        pass


class BaseSLModel(BaseModel):

    def __init__(self, model_name, x_space, y_space, **options):
        super(BaseSLModel, self).__init__(model_name, x_space, y_space, **options)

    def train(self, x_train, y_train, x_validate, y_validate):
        # 1. Last evaluate loss for early stop.
        loss_evaluate_min, loss_evaluate_min_step = np.inf, 0
        # 2. Train loop.
        for train_step in range(self.train_steps_limit):
            # 2.1. Get data_handler size.
            data_size = len(x_train)
            # 2.2. Get mini batch.
            indices = np.random.choice(data_size, size=self.batch_size_train)
            x_batch = x_train[indices]
            y_batch = y_train[indices]
            # 2.3. Train op.
            ops = [self.loss_func, self.optimizer]
            if self.train_steps % self.train_save_steps == 0:
                ops.append(self.summary_merge_all_op)
            # 2.4. Train.
            results = self.session.run(ops, {
                self.x_input: x_batch,
                self.y_input: y_batch,
                self.t_dropout_keep_prob: self.dropout_prob,
                self.t_is_training: True
            })
            loss_train = results[0]
            # 2.5. Add summary & Save model & Validation & Early-stop.
            if self.train_steps % self.train_save_steps == 0:
                # 2.5.1 Save summary.
                self.summary_writer.add_summary(results[-1], global_step=self.train_steps)
                # 2.5.2 Evaluate loss.
                loss_evaluate = self.evaluate(x_validate, y_validate)
                self.logger.warning('Steps: {0} | Train loss: {1:.8f} | Validate Loss: {2:.8f}'.format(self.train_steps,
                                                                                                       loss_train,
                                                                                                       loss_evaluate))
                if loss_evaluate < loss_evaluate_min:
                    # 2.5.3 Save model & graph.
                    self.save(need_save_graph=True)
                    # 2.5.4 Update min evaluate loss.
                    loss_evaluate_min = loss_evaluate
                    # 2.5.5 Update min evaluate loss step.
                    loss_evaluate_min_step = train_step
                else:
                    if train_step - loss_evaluate_min_step >= self.train_save_steps:
                        self.logger.warning('Validate loss does not decrease for {} steps, early stop.'.format(self.train_save_steps))
                        break

            # 6. Train steps ++.
            self.train_steps += 1

    def predict(self, x):
        y_predict = self.session.run(self.y_predict, {self.x_input: x})
        return y_predict

    def evaluate(self, x_input, y_input):
        loss = []
        # 1. Get data_handler count.
        data_count = len(x_input)
        # 2. Calculate batch count.
        batch_count = int(math.ceil(float(data_count) / self.batch_size_evaluate))
        # 3. Evaluate.
        for batch_index in range(batch_count):
            l_bound, r_bound = batch_index * self.batch_size_evaluate, (batch_index + 1) * self.batch_size_evaluate
            _predict, _loss = self.session.run([self.y_predict, self.loss_func], {
                self.x_input: x_input[l_bound: r_bound],
                self.y_input: y_input[l_bound: r_bound]
            })
            loss.append(_loss)
        loss = np.array(loss).mean()
        return loss
