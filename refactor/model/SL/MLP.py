# coding=utf-8

import tensorflow as tf
import numpy as np
import math

from base.model import BaseSLModel


class Agent(BaseSLModel):

    def _init_input(self, *args):
        self.x_input = tf.placeholder(tf.float32, [None, self.x_space], name='x_input')
        self.y_input = tf.placeholder(tf.float32, [None, self.y_space], name='y_input')
        self.t_dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.t_is_training = tf.placeholder_with_default(False, shape=())

    def _init_nn(self, *args):
        with tf.variable_scope('MLP'):
            # 1. Encoder layers.
            w_initializer = tf.random_normal_initializer(0.0, 0.0001)
            f1_dense = tf.layers.dense(self.x_input, 256, tf.nn.selu, kernel_initializer=w_initializer)
            f1_dense = tf.layers.batch_normalization(f1_dense, training=self.t_is_training)
            f1_dense = tf.layers.dropout(f1_dense, self.t_dropout_keep_prob)
            f2_dense = tf.layers.dense(f1_dense, 512, tf.nn.selu, kernel_initializer=w_initializer)
            f2_dense = tf.layers.batch_normalization(f2_dense, training=self.t_is_training)
            f2_dense = tf.layers.dropout(f2_dense, self.t_dropout_keep_prob)
            f_dense = tf.layers.dense(f2_dense, 128, tf.nn.selu, kernel_initializer=w_initializer)
            f_dense = tf.layers.batch_normalization(f_dense, training=self.t_is_training)
            f_dense = tf.layers.dropout(f_dense, self.t_dropout_keep_prob)
            # 2. Output layers.
            if self.loss_func_name == 'MSE':
                self.y_predict = tf.layers.dense(f_dense, self.y_space, name='y_predict')
                self.y_predict = tf.layers.batch_normalization(self.y_predict, training=self.t_is_training)
            else:
                self.y_logits = tf.layers.dense(f_dense, self.y_space)
                self.y_predict = tf.nn.softmax(self.y_logits, name='y_predict')

    def _init_op(self):
        with tf.variable_scope('loss_func'):
            if self.loss_func_name == 'MSE':
                self.loss_func = tf.losses.mean_squared_error(self.y_input, self.y_predict)
                tf.summary.scalar('mse', self.loss_func)
            elif self.loss_func_name == 'CE':
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_input, logits=self.y_logits)
                self.loss_func = tf.reduce_mean(cross_entropy)
                tf.summary.scalar('ce', self.loss_func)
            else:
                raise NotImplementedError
        with tf.variable_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_func)

    def _init_signature(self):
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
        self.session.run(tf.global_variables_initializer())
