# coding=utf-8

import tensorflow as tf
import shutil
import os

import config

from utility import logger
from abc import abstractmethod


class BaseModel(object):

    def __init__(self, model_name, **options):
        # 1. Model related.
        self.model_name = model_name
        # 2. Training related.
        self._init_options(options)
        # 3. Saving related.
        self.model_dir = os.path.join(config.SAVED_MODEL_DIR, config.DATETIME_LOAD_MODULE, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # 3.1 Saver.
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoint')
        self.checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        self.saver = tf.train.Saver()
        # 3.2 Summary Writer.
        self.summary_dir = os.path.join(self.model_dir, 'summaries')
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        # TODO - Session init before this.
        self.summary_writer = tf.summary.FileWriter(self.summary_dir, graph=self.session.graph)
        self.summary_merge_all_op = tf.summary.merge_all()
        # 3.3 Graph builder.
        self.graph_dir = os.path.join(self.model_dir, 'graph')
        self.builder_signature = None
        # 4. Logger.
        self.logger = logger.get_logger(model_name)

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

    def __init__(self, a_space, s_space, **options):
        super(BaseRLModel, self).__init__(**options)
        # Init spaces for action and state.
        self.a_space, self.s_space = a_space, s_space
        # Init buffer count.
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

    def __init__(self, x_space, y_space, **options):
        super(BaseSLModel, self).__init__(**options)
        # Feature space and Label space.
        self.x_space, self.y_space = x_space, y_space

