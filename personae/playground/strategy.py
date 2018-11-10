# coding=utf-8

from personae.contrib.data.handler import PredictorDataHandler
from personae.contrib.model.model import LightGBMModel, MLPModel

data_handler = PredictorDataHandler(raw_data_dir=r'/Users/shuyu/Desktop/Affair/Temp/data_tmp/processed',
                                    normalize_data=True,
                                    drop_nan_columns=True)
# data_handler = PredictorDataHandler(raw_data_dir=r'D:\Users\v-shuyw\data\ycz\data_sample\processed')

# model = LightGBMModel(**{
#     'num_threads': 4,
# })
#

x_space, y_space = data_handler.x_train.shape[1], 1

model = MLPModel(x_space=x_space, y_space=y_space)

model.fit(data_handler.x_train,
          data_handler.y_train.reshape((-1, 1)),
          data_handler.x_validation,
          data_handler.y_validation.reshape((-1, 1)))


