#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
    Visualize the loss to see its distribution among time axis and space axis
"""

import time
import torch, gc
import numpy as np
import d2lzh_pytorch as d2l
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
from data.PeMS import load_pems_spd_data, load_mobile_century_spd_data
import pandas as pd
from stgcn import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, EarlyStopping, ChebPolynomial

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    '''
        当epoch设置太长的时候，可以在发现效果已经很难看的时候马上中止训练，然后把各项误差copy出来在这里作图
    '''
    # plt.figure()
    # plt.plot(training_losses, label="training loss")
    # plt.plot(validation_losses, label="validation loss")
    # plt.title("Loss Curve")
    # plt.legend()
    # plt.figure()
    # plt.plot(validation_maes, label="validation mae")
    # plt.plot(validation_mapes, label="validation mape(%)")
    # plt.title("Validation MAEs and MAPEs")
    # plt.legend()
    # plt.show()
    """
        make sure test_batch size 能够整除 test_input.shape[0]
        一种稳妥的方式是直接取sample_per_day(考虑到训练验证划分是整天整天分的，因此样本数肯定是每天样本数的整数倍)
    """
    num_timesteps_input = 12
    num_timesteps_output = 3
    FORECAST_GAP = 2
    # num_detector = 129
    data_per_day = 288
    sample_per_day = data_per_day - num_timesteps_input - (FORECAST_GAP + 1) * num_timesteps_output + 1
    test_batchsize = sample_per_day

    A, means, stds, _, _, valid_input, valid_target, test_input, test_target = \
        load_mobile_century_spd_data(input_steps=num_timesteps_input, output_steps=num_timesteps_output, FORECAST_GAP=FORECAST_GAP,with_history=1)
    print("valid data shape: {}, test data shape: {}".format(valid_input.shape[:], test_input.shape[:]))

    Ks = 3
    Kt = 3
    #  A: Adjacent Matrix. here A_Wave == D_wave*(A+I)*D_wave.T
    A_wave = get_normalized_adj(A)
    Cheb_Poly = ChebPolynomial(A_wave, Ks).to(device=device)
    # num_nodes, num_features, num_timesteps_input, num_timesteps_output
    net = STGCN(A_wave.shape[0],
                test_input.shape[3],
                num_timesteps_input,
                num_timesteps_output, Ks=Ks, Kt=Kt, ChebPoly=Cheb_Poly).to(device=device)
    net.eval()
    net.load_state_dict(torch.load('checkpoint.pt'))
    print("trained model loaded")

    # loss_criterion = nn.MSELoss()

    test_input = test_input.to(device=device)
    test_target = test_target.to(device=device)

    # validate set is too large to be passed to cuda all at once
    loss, mae, mape = 0.0, 0.0, 0.0

    for i in range(0, test_input.shape[0], test_batchsize):
        gc.collect()
        torch.cuda.empty_cache()  # at the beginning of each train, clear the unused cache

        j = i + test_batchsize
        test_X_batch = test_input[i:j]
        test_y_batch = test_target[i:j]
        # batch_ = test_X_batch.shape[0]  # real batch size(last few samples mod(test_input.shape[0], test_batch_size))
        out = net(test_X_batch)
        with torch.no_grad():
            loss += ((out - test_y_batch) ** 2).mean(axis=0).detach().cpu().numpy()

            # inverse-normalization
            out_unnormalized = out.detach().cpu().numpy() * stds[0] + means[0]
            target_unnormalized = test_y_batch.detach().cpu().numpy() * stds[0] + means[0]
            mae += np.mean(np.absolute(out_unnormalized - target_unnormalized), axis=0)

            # avoid divide by 0
            target_unnormalized_nonzero = np.copy(target_unnormalized)
            target_unnormalized_nonzero[target_unnormalized_nonzero[:] == 0] = means[0]
            mape += np.mean(np.absolute(target_unnormalized - out_unnormalized) / target_unnormalized_nonzero, axis=0)

            out = None

    loss /= (test_input.shape[0] // test_batchsize)
    mae /= (test_input.shape[0] // test_batchsize)
    mape /= (test_input.shape[0] // test_batchsize)

    test_mse = np.mean(loss)
    test_mae = np.mean(mae)
    test_mape = np.mean(mape)

    mean_time_mae = np.mean(mae, axis=0)
    mean_detector_mae = np.mean(mae, axis=1)
    mean_time_mse = np.mean(loss*stds**2, axis=0)
    mean_detector_mae = np.mean(loss*stds**2, axis=1)
    mean_time_mape = np.mean(mape, axis=0)
    mean_detector_mape = np.mean(mape, axis=1)


    # plot
    plt.figure()
    plt.plot(mean_time_mae)
    plt.ylabel('MAEs', fontsize=18)
    plt.xlabel('forecast steps', fontsize=18)

    plt.figure()
    plt.plot(mean_detector_mae)
    plt.axhline(y=mean_detector_mae.mean(), ls='--', color='r')
    plt.ylabel('MAEs', fontsize=18)
    plt.xlabel('detector', fontsize=18)

    fig, ax0 = plt.subplots()
    c = ax0.pcolor(loss, cmap='Greys')  # cmap: name of chosen colormap, see
    # https://matplotlib.org/3.5.3/tutorials/colors/colormaps.html
    ax0.set_title(f'Speed Prediction MSE', fontsize=20)
    plt.ylabel('detectors', fontsize=18)
    plt.xlabel('forecast steps', fontsize=18)

    fig, ax0 = plt.subplots()
    c = ax0.pcolor(mae, cmap='Greys')  # cmap: name of chosen colormap, see
    # https://matplotlib.org/3.5.3/tutorials/colors/colormaps.html
    ax0.set_title(f'Speed Prediction MAE', fontsize=20)
    plt.ylabel('detectors', fontsize=18)
    plt.xlabel('forecast steps', fontsize=18)

    fig, ax0 = plt.subplots()
    c = ax0.pcolor(mape, cmap='Greys')  # cmap: name of chosen colormap, see
    # https://matplotlib.org/3.5.3/tutorials/colors/colormaps.html
    ax0.set_title(f'Speed Prediction MAPE', fontsize=20)
    plt.ylabel('detectors', fontsize=18)
    plt.xlabel('forecast steps', fontsize=18)
    # ticks = [0, 24, 48, 72, 96, 120, 144, 168, 192]
    # labels = ['5:00', '7:00', '9:00', '11:00', '13:00', '15:00', '17:00', '19:00', '21:00']
    # plt.yticks(ticks=ticks, labels=labels, fontsize=16)
    # plt.xticks(fontsize=18)
    fig.tight_layout()

    plt.show()
    print('')
