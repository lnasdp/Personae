# coding=utf-8

import tensorflow as tf
import lightgbm as gbm
import numpy as np

from personae.contrib.data.handler import PredictorDataHandler
# from personae.contrib.model.modeml import LightGBMModel
from personae.contrib.model.model import MLPModel

# data_handler = PredictorDataHandler(processed_data_dir=r'/Users/shuyu/Desktop/Affair/Temp/data_tmp/processed',
#                                     normalize_data=True,)
# data_handler = PredictorDataHandler(processed_data_dir=r'D:\Users\v-shuyw\data\ycz\data_sample\processed',
#                                     normalize_data=True)

data_handler = PredictorDataHandler(processed_data_dir=r'D:\Users\v-shuyw\data\ycz\data\processed',
                                    normalize_data=False)


# model = LightGBMModel(**{
#     'num_threads': 4,
# })
#

x_space, y_space = data_handler.x_train.shape[1], 1

# model = MLPModel(x_space=x_space, y_space=y_space, train_steps=500)
#
# model.fit(data_handler.x_train.values,
#           data_handler.y_train.values.reshape((-1, 1)),
#           data_handler.x_validation.values,
#           data_handler.y_validation.values.reshape((-1, 1)))
#
# model.load()


# sess = tf.Session()
#
# x = tf.placeholder(tf.float32, [None, x_space])
# y = tf.placeholder(tf.float32, [None, y_space])
#
# l1 = tf.layers.dense(x, 256, activation=tf.nn.relu)
# l2 = tf.layers.dense(l1, 256, activation=tf.nn.relu)
# l3 = tf.layers.dense(l2, 256, activation=tf.nn.relu)
#
# p = tf.layers.dense(l3, 1)
#
# loss = tf.losses.mean_squared_error(y, p)
#
# train = tf.train.AdamOptimizer().minimize(loss)
#
# sess.run(tf.global_variables_initializer())
#
# for t in range(1000):
#     sess.run(train, {
#         x: data_handler.x_train.values,
#         y: data_handler.y_train.values.reshape((-1, 1))
#     })
#
# print(np.corrcoef(data_handler.y_train, (sess.run(p, {x: data_handler.x_train.values})).reshape((-1, ))))
# print(np.corrcoef(data_handler.y_validation, (sess.run(p, {x: data_handler.x_validation.values})).reshape((-1, ))))
# print(np.corrcoef(data_handler.y_test, (sess.run(p, {x: data_handler.x_test.values})).reshape((-1, ))))

# print(np.corrcoef(data_handler.y_train, model.predict(data_handler.x_train.values).reshape((-1, ))))
# print(np.corrcoef(data_handler.y_validation, model.predict(data_handler.x_validation.values).reshape((-1, ))))
# print(np.corrcoef(data_handler.y_test, model.predict(data_handler.x_test.values).reshape((-1, ))))


train_set = gbm.Dataset(data_handler.x_train.values, label=data_handler.y_train.values)
validation_set = gbm.Dataset(data_handler.x_validation.values, label=data_handler.y_validation.values)

eval_result = dict()

model = gbm.train(
    params={'num_threads': 20,
            'objective': 'mse'},
    train_set=train_set,
    verbose_eval=50,
    valid_sets=[train_set, validation_set],
    evals_result=eval_result,
    num_boost_round=1000,
    early_stopping_rounds=50
)

print(np.corrcoef(data_handler.y_train, model.predict(data_handler.x_train.values)))
print(np.corrcoef(data_handler.y_validation, model.predict(data_handler.x_validation.values)))
print(np.corrcoef(data_handler.y_test, model.predict(data_handler.x_test.values)))
