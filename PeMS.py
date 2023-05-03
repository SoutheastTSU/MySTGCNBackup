#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import math


def create_dataset_with_avg_historical(data, input_steps, output_steps, num_day, GAP=0):
    """
        data: shape(channel, time(192 per day), nodes)
        only historical data of previous day is considered
        forecast X(t+3),X(t+6),X(t+9) using X(t-input_steps)~X(t-1) and X(t-day-shift)~X(t-day-shift+input_steps-1)
    """
    shift = 3  # time shift of historical data
    avg = historical_average_MoblieCentury()

    # data: shape(channel, time(192 per day), nodes) -> (nodes, channel, time)
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 0, 1)
    data_per_day = data.shape[2] // num_day

    # generate dataset
    dataset_x, dataset_y = [], []
    for j in range(num_day):
        base_idx = j * data_per_day
        for i in range(data_per_day - input_steps - (1 + GAP) * output_steps + 1):
            X = data[:, :, base_idx + i:base_idx + i + input_steps]
            X_his = avg[i + input_steps - shift: i + input_steps + (1 + GAP) * output_steps, :]
            X_his = X_his.reshape(X_his.shape[1], 1, X_his.shape[0])
            if X.shape != X_his.shape:
                print("")
            X = np.append(X, X_his, 1)  # 1st channel: past input_steps data, 2st channel: history avg data
            out_idx = base_idx + i + input_steps - 1
            y = []
            while out_idx < base_idx + i + input_steps - 1 + (1 + GAP) * output_steps:
                out_idx += 1 + GAP
                y.append(data[:, :, out_idx].reshape(-1))
            y = np.array(y).T
            dataset_x.append(X)
            dataset_y.append(y)

    # list to tensor
    # shape: batch, channel, row(detector), column(time), (meaning of channel is the same as pics)
    dataset_x = torch.from_numpy(np.array(dataset_x))
    # shape: batch, row(detector), column
    dataset_y = torch.from_numpy(np.array(dataset_y))
    return dataset_x, dataset_y


def historical_average():
    data_per_day = 192  # 5-min interval data
    # sample_per_day = data_per_day - input_steps - (1 + FORECAST_GAP) * output_steps + 1

    TRAIN_DATA = np.load(
        r"F:\Graduate\AverageSpeedPrediction\CNN\PeMS_Data\Self_Downloaded\PeMSSpd\2014040506__521time_129det_weekday.npz")
    TRAIN_DATA = TRAIN_DATA['data']
    TEST_DATA = np.load(
        r"F:\Graduate\AverageSpeedPrediction\CNN\PeMS_Data\Self_Downloaded\PeMSSpd\1407_521time_129det_weekday.npz")
    TEST_DATA = TEST_DATA['data']

    FULL_DATA = np.append(TRAIN_DATA, TEST_DATA, axis=0).astype('float32')  # shape: sample * num_detector
    FULL_DATA = np.delete(FULL_DATA, [3, 12, 117, 123, 124, 126], axis=1)  # 这一行是手动删除数据里面idx=117的探测点1201222

    full_mean = np.mean(FULL_DATA)
    full_std = np.std(FULL_DATA)

    # train_data: shape=[channel, time, detector]
    timesteps, num_detector = FULL_DATA.shape
    num_day = timesteps // data_per_day
    avg, square_avg = None, None
    # calculates avg and stds on time-of-the-day(计算查看不同天数在同一时刻上的速度值的变化幅度，评估历史数据的模式固定程度)
    for i in range(num_day):
        tmp_slice = slice(i * data_per_day, (i + 1) * data_per_day)
        tmp = FULL_DATA[tmp_slice, :]
        if i == 0:
            avg = tmp
            square_avg = tmp ** 2
        else:
            avg += tmp
            square_avg += tmp ** 2
    avg /= num_day
    # square_avg /= num_day
    # std = np.sqrt(square_avg - avg**2)
    # stds = np.mean(std, axis=0)
    # plt.plot(stds)
    # plt.show()
    # print("")
    return (avg - full_mean) / full_std


def historical_average_MoblieCentury():
    data_per_day = 288
    FULL_DATA = np.load(r"F:\Graduate\AverageSpeedPrediction\MobileCentury\dataRaw\KLunder015Det76.npz")
    FULL_DATA = FULL_DATA['data']
    full_mean = np.mean(FULL_DATA)
    full_std = np.std(FULL_DATA)

    # train_data: shape=[channel, time, detector]
    timesteps, num_detector = FULL_DATA.shape
    num_day = timesteps // data_per_day
    avg, square_avg = None, None
    # calculates avg and stds on time-of-the-day(计算查看不同天数在同一时刻上的速度值的变化幅度，评估历史数据的模式固定程度)
    for i in range(num_day):
        tmp_slice = slice(i * data_per_day, (i + 1) * data_per_day)
        tmp = FULL_DATA[tmp_slice, :]
        if i == 0:
            avg = tmp
            square_avg = tmp ** 2
        else:
            avg += tmp
            square_avg += tmp ** 2
    avg /= num_day
    # square_avg /= num_day
    # std = np.sqrt(square_avg - avg**2)
    # stds = np.mean(std, axis=0)
    # plt.plot(stds)
    # plt.show()
    # print("")
    return (avg - full_mean) / full_std


def create_dataset(data, input_steps, output_steps, num_day, GAP=0):
    # 数据并不连贯，每天只有固定时段内的数据。
    # 输入为input_steps时间长度的数据，输出为output_steps时间长度。
    dataset_x, dataset_y = [], []

    # train_data: shape=[channel, time, detector]
    num_channel, timesteps, num_detector = data.shape
    data_per_day = timesteps // num_day
    # temp = np.copy(data)
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 0, 1)

    for j in range(num_day):
        base_idx = j * data_per_day
        for i in range(data_per_day - input_steps - (1 + GAP) * output_steps + 1):
            X = data[:, :, base_idx + i:base_idx + i + input_steps]
            out_idx = base_idx + i + input_steps - 1
            y = []
            while out_idx < base_idx + i + input_steps - 1 + (1 + GAP) * output_steps:
                out_idx += 1 + GAP
                y.append(data[:, :, out_idx].reshape(-1))
            y = np.array(y).T
            dataset_x.append(X)
            dataset_y.append(y)

    # list to tensor
    # shape: batch, channel, row(detector), column(time), (meaning of channel is the same as pics)
    dataset_x = torch.from_numpy(np.array(dataset_x))
    # shape: batch, row(detector), column
    dataset_y = torch.from_numpy(np.array(dataset_y))
    return dataset_x, dataset_y


def load_pems_spd_data(input_steps=12, output_steps=3, FORECAST_GAP=2, with_history=0):
    data_per_day = 192  # 5-min interval data
    sample_per_day = data_per_day - input_steps - (1 + FORECAST_GAP) * output_steps + 1

    TRAIN_DATA = np.load(
        r"F:\Graduate\AverageSpeedPrediction\CNN\PeMS_Data\Self_Downloaded\PeMSSpd\2014040506__521time_129det_weekday.npz")
    TRAIN_DATA = TRAIN_DATA['data']
    TEST_DATA = np.load(
        r"F:\Graduate\AverageSpeedPrediction\CNN\PeMS_Data\Self_Downloaded\PeMSSpd\1407_521time_129det_weekday.npz")
    TEST_DATA = TEST_DATA['data']

    FULL_DATA = np.append(TRAIN_DATA, TEST_DATA, axis=0).astype('float32')  # shape: sample * num_detector
    FULL_DATA = np.delete(FULL_DATA, [3, 12, 117, 123, 124, 126], axis=1)  # 这一行是手动删除数据里面idx=117的探测点1201222
    full_mean = np.mean(FULL_DATA)
    full_std = np.std(FULL_DATA)
    Normalized = (FULL_DATA - full_mean) / full_std
    print(Normalized.max(), Normalized.min())
    if not with_history:
        X, y = create_dataset(data=Normalized.reshape(1, Normalized.shape[0], Normalized.shape[1]),
                              input_steps=input_steps, output_steps=output_steps,
                              num_day=Normalized.shape[0] // data_per_day,
                              GAP=FORECAST_GAP)  # shape: sample, detector, channel, timesteps
    else:
        X, y = create_dataset_with_avg_historical(data=Normalized.reshape(1, Normalized.shape[0], Normalized.shape[1]),
                                                  input_steps=input_steps, output_steps=output_steps,
                                                  num_day=Normalized.shape[0] // data_per_day,
                                                  GAP=FORECAST_GAP)  # shape: sample, detector, channel, timesteps
    means = np.array(full_mean).reshape(-1)
    std = np.array(full_std).reshape(-1)

    # total 81 days, 65+16, 65:5 fold
    train_days = 48
    valid_days = 16 + train_days
    test_days = 19 + valid_days

    training_input = X[0:sample_per_day * train_days]
    valid_input = X[sample_per_day * train_days:sample_per_day * valid_days]
    test_input = X[sample_per_day * valid_days:sample_per_day * test_days]

    training_input = np.swapaxes(training_input, 2, 3)
    valid_input = np.swapaxes(valid_input, 2, 3)
    test_input = np.swapaxes(test_input, 2, 3)

    training_target = y[0:sample_per_day * train_days, :, :]
    valid_target = y[sample_per_day * train_days:sample_per_day * valid_days, :, :]
    test_target = y[sample_per_day * valid_days:sample_per_day * test_days, :, :]

    A = Adjacent_Matrix()

    return A, means, std, \
           training_input, training_target, \
           valid_input, valid_target, \
           test_input, test_target


def load_mobile_century_spd_data(input_steps=12, output_steps=3, FORECAST_GAP=2, with_history=0):
    """
        MobileCenturyData, 76 detectors in 20070201-20080229
    """
    data_per_day = 288  # 5-min interval data
    sample_per_day = data_per_day - input_steps - (1 + FORECAST_GAP) * output_steps + 1

    FULL_DATA = np.load(r"F:\Graduate\AverageSpeedPrediction\MobileCentury\dataRaw\KLunder015Det76.npz")
    FULL_DATA = FULL_DATA['data']

    full_mean = np.mean(FULL_DATA)
    full_std = np.std(FULL_DATA)
    Normalized = (FULL_DATA - full_mean) / full_std
    print(Normalized.max(), Normalized.min())
    if not with_history:
        X, y = create_dataset(data=Normalized.reshape(1, Normalized.shape[0], Normalized.shape[1]),
                              input_steps=input_steps, output_steps=output_steps,
                              num_day=Normalized.shape[0] // data_per_day,
                              GAP=FORECAST_GAP)  # shape: sample, detector, channel, timesteps
    else:
        X, y = create_dataset_with_avg_historical(data=Normalized.reshape(1, Normalized.shape[0], Normalized.shape[1]),
                                                  input_steps=input_steps, output_steps=output_steps,
                                                  num_day=Normalized.shape[0] // data_per_day,
                                                  GAP=FORECAST_GAP)  # shape: sample, detector, channel, timesteps
    means = np.array(full_mean).reshape(-1)
    std = np.array(full_std).reshape(-1)

    # total 81 days, 65+16, 65:5 fold
    train_days = 150
    valid_days = 50 + train_days
    test_days = 48 + valid_days

    training_input = X[0:sample_per_day * train_days]
    valid_input = X[sample_per_day * train_days:sample_per_day * valid_days]
    test_input = X[sample_per_day * valid_days:sample_per_day * test_days]

    training_input = np.swapaxes(training_input, 2, 3)
    valid_input = np.swapaxes(valid_input, 2, 3)
    test_input = np.swapaxes(test_input, 2, 3)

    training_target = y[0:sample_per_day * train_days, :, :]
    valid_target = y[sample_per_day * train_days:sample_per_day * valid_days, :, :]
    test_target = y[sample_per_day * valid_days:sample_per_day * test_days, :, :]

    A = Adjacent_Matrix_MobileCentury()

    return A, means, std, \
           training_input, training_target, \
           valid_input, valid_target, \
           test_input, test_target


def Adjacent_Matrix():  # 只面向首尾相连的道路,lst中的元素代表到下一个点的距离（最后以0结尾）
    mile_to_meter = 1.609344 * 1000
    npz = np.load(r"F:\Graduate\AverageSpeedPrediction\CNN\PeMS_Data\Self_Downloaded\PeMSSpd\detector_postmile.npz")
    data = npz['data']
    data = np.delete(data, [3, 12, 117, 123, 124, 126], axis=0)  # 这一行是手动删除数据里面idx=117的探测点1201222
    # [72.07, 71.71, 69.97, 69.47, 68.64, 67.64, 66.97, 66.23, 65.25, 64.67, 64.10, 63.77, 62.51, 61.91, 61.34,
    #    60.70,
    #    59.57, 58.48, 57.19, 56.73, 56.17, 55.34, 54.72, 54.24, 53.90, 53.27, 52.93, 52.31, 52.29, 51.87, 51.58,
    #    51.12,
    #    50.61, 49.92, 49.77, 49.18, 48.87, 48.02, 47.38, 47.24, 47.13, 46.47, 46.45, 46.11, 45.14, 44.37, 44.16,
    #    43.90,
    #    43.34, 43.13, 42.93, 42.12, 41.97, 41.29, 40.78, 40.43, 39.77, 39.20, 38.73, 38.11, 37.58, 37.08, 36.95,
    #    36.34,
    #    35.82, 35.59, 35.09, 34.67, 34.43, 33.24, 32.86, 32.46, 32.22, 31.79, 31.40, 30.99, 30.55, 30.33, 30.04,
    #    29.76,
    #    29.41, 28.87, 28.76, 28.52, 28.16, 27.77, 27.39, 27.23, 26.69, 26.40, 25.68, 25.05, 24.55, 24.06, 23.89,
    #    23.77,
    #    23.77, 23.39, 23.32, 22.89, 22.32, 21.68, 21.33, 20.65, 20.10, 19.41, 19.01, 18.42, 17.69, 17.22, 16.53,
    #    16.29,
    #    15.64, 15.16, 14.94, 14.59, 14.31, 13.74, 13.51, 12.93, 12.62, 11.93, 11.37, 11.17, 10.67, 9.87, 9.67, 9.42,
    #    9.23, 8.97, 8.97, 8.47, 8.17, 8.03, 7.50, 6.84, 6.62, 5.98, 5.51, 5.32, 4.82, 4.78, 3.80, 3.63, 3.08, 2.81,
    #    2.66, 2.12, 1.70, 1.34, 0.88, 0.70, 0.37]
    # lst = np.array(lst)
    # detector_drop_out_list = [27, 41, 97, 129, 140]  # 0-indexed index of detectors not used
    # data = np.delete(lst, detector_drop_out_list)  # delete un-used data
    # data = data[21:]
    data = data * mile_to_meter  # throw out 0-20 detectors
    distance = data[0:data.shape[0] - 1] - data[1:]
    distance = np.append(distance, [0])

    sigma_2 = 1500 * 2
    epsilon = 0.1
    # A = np.zeros((distance.shape[0], distance.shape[0]), dtype='float32')
    A = np.eye(distance.shape[0], dtype='float32')
    for i in range(A.shape[0] - 1):
        A[i, i + 1] = math.exp(- distance[i] ** 2 / sigma_2)
        A[i + 1, i] = A[i, i + 1]
    A[A[:] < epsilon + 10 ** -3] = 0

    # # 查看A的值是否合理，判断sigma和epsilon取值是否合理
    # temp=[]
    # for i in range(A.shape[0] - 1):
    #     temp.append(A[i, i+1])
    # temp = np.array(temp)
    # plt.plot(temp)
    # plt.show()
    return A


def Adjacent_Matrix_MobileCentury():
    sigma_2 = 1500 * 2
    epsilon = 0.1
    detNum = 76
    # A = np.zeros((distance.shape[0], distance.shape[0]), dtype='float32')
    A = np.zeros((detNum, detNum), dtype='float32')
    for i in range(A.shape[0] - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A


if __name__ == '__main__':
    # load_pems_spd_data(input_steps=12, output_steps=3, FORECAST_GAP=2, with_history=1)
    load_mobile_century_spd_data(input_steps=12, output_steps=3, FORECAST_GAP=2, with_history=1)
    load_mobile_century_spd_data(input_steps=12, output_steps=3, FORECAST_GAP=2, with_history=0)
    # historical_average()
    print('')
