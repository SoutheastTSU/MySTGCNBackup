import time
import torch, gc
import numpy as np
import d2lzh_pytorch as d2l
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import pandas as pd

# 和其他方法一样的输出：考虑output_steps下的平均误差，shape最好是[num_sample, num_detector, output_steps]
pems = np.load(
    r"F:\Graduate\AverageSpeedPrediction\CNN\LeNet\14040506_Clipped_WithTimestamp_OnlyWeekday.npz")
data = pems['data']
num_day = 59
print("Shape of test data is", data.shape)

sample_per_day = 192
num_detector = 127
HA_value = np.zeros((sample_per_day, num_detector))
for i in range(num_day):
    sl = slice(i * sample_per_day, (i + 1) * sample_per_day)
    HA_value += data[1, sl, :]
HA_value /= num_day

# test_x = np.load(r"F:\Graduate\AverageSpeedPrediction\CNN\LeNet\test_x.npz")
# test_x = test_x['data']
# test_x = torch.tensor(test_x[:, 1, :, :].reshape(test_x.shape[0], 1, -1, test_x.shape[-1]))  # timestamp not considered
# test_y = np.load(r"F:\Graduate\AverageSpeedPrediction\CNN\LeNet\test_y.npz")
# test_y = torch.tensor(test_y['data'])
# num_test_day = 22
test_data = np.load(
    r"F:\Graduate\AverageSpeedPrediction\CNN\LeNet\1407_Clipped_WithTimestamp_OnlyWeekday.npz")
test_data = test_data['data']
print("Shape of test data is", test_data.shape)
test_data = test_data[1, :, :]

test_day = 22
error, square_error, percentage_error = 0.0, 0.0, 0.0
for i in range(test_day):
    sl = slice(i * sample_per_day, (i + 1) * sample_per_day)
    temp = abs(HA_value - test_data[sl, :])
    error += temp
    square_error += temp**2
    percentage_error += temp/test_data[sl, :]
error /= test_day
square_error /= test_day
percentage_error /= test_day

MSE = np.mean(square_error[24:, :])
MAE = np.mean(error[24:, :])
MAPE = np.mean(percentage_error[24:, :])
print(f"mae: {MAE}, mse:{MSE}, mape:{MAPE}")
