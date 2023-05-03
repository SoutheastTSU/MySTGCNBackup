import os
import argparse
import pickle as pk
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import torch.nn as nn
from stgcn import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, EarlyStopping
from data.PeMS import load_pems_spd_data
# from BPNN import train_epoch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_evaluate(net, test_input, test_target):
    net.eval()
    test_batchsize = sample_per_day

    test_input = test_input.to(device=device)
    test_target = test_target.to(device=device)

    # validate set is too large to be passed to cuda all at once
    loss, mae, mape = 0.0, 0.0, 0.0
    with torch.no_grad():
        for i in range(0, test_input.shape[0], test_batchsize):
            test_X_batch = test_input[i:i + test_batchsize]
            test_y_batch = test_target[i:i + test_batchsize]
            out = net(test_X_batch)
            loss += ((out - test_y_batch) ** 2).mean(axis=0).detach().cpu().numpy()

            # inverse-normalization
            out_unnormalized = out.detach().cpu().numpy() * stds[0] + means[0]
            target_unnormalized = test_y_batch.detach().cpu().numpy() * stds[0] + means[0]
            mae += np.mean(np.absolute(out_unnormalized - target_unnormalized), axis=0)

            # avoid divide by 0
            target_unnormalized_nonzero = np.copy(target_unnormalized)
            target_unnormalized_nonzero[target_unnormalized_nonzero[:] == 0] = means[0]
            mape += np.mean(np.absolute(target_unnormalized - out_unnormalized) / target_unnormalized_nonzero, axis=0)

    loss /= (test_input.shape[0] // test_batchsize)
    mae /= (test_input.shape[0] // test_batchsize)
    mape /= (test_input.shape[0] // test_batchsize)

    test_mse = np.mean(loss)
    test_mae = np.mean(mae)
    test_mape = np.mean(mape)

    mean_time_mae = np.mean(mae, axis=0)
    # mean_detector_mae = np.mean(mae, axis=1)
    mean_time_mse = np.mean(loss * stds ** 2, axis=0)
    # mean_detector_mae = np.mean(loss * stds ** 2, axis=1)
    mean_time_mape = np.mean(mape, axis=0)
    # mean_detector_mape = np.mean(mape, axis=1)

    print('mse: {}, mae: {}, mape: {}'.format(mean_time_mse, mean_time_mae, mean_time_mape))

    return test_mse, test_mae, test_mape


class Classic_CNN(nn.Module):
    def __init__(self, num_timesteps_input, num_timesteps_output, num_detector):
        super(Classic_CNN, self).__init__()
        self.input_steps = num_timesteps_input
        self.output_steps = num_timesteps_output
        self.num_detector = num_detector
        self.input_channel = 2
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Conv2d(self.input_channel, 256, 3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2, padding=1),  # kernel_size, stride
            nn.Conv2d(256, 128, 3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (((self.num_detector + 2) // 2 + 2) // 2 + 2) // 2 * (
                        ((self.input_steps + 2) // 2 + 2) // 2 + 2) // 2, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(2048, self.output_steps * self.num_detector)
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(2048, self.output_steps * self.num_detector)
        )

    def forward(self, img):
        # img: N, det, step, channel(NHWC), transform to NCHW for conv
        img = img.permute(0, 3, 1, 2)
        batch_ = img.shape[0]
        img = self.conv(img)
        img = self.fc(img.view(batch_, -1))  # flatten to vectors
        img = img.view(batch_, self.num_detector, self.output_steps)
        return img


class ImprovedFC_CNN(nn.Module):
    def __init__(self, num_timesteps_input, num_timesteps_output, num_detector, input_channel):
        super(ImprovedFC_CNN, self).__init__()
        self.input_steps = num_timesteps_input
        self.output_steps = num_timesteps_output
        self.num_detector = num_detector
        self.input_channel = input_channel
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Conv2d(self.input_channel, 256, 3, padding=(1, 0)),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, padding=(1, 0)),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.fc = nn.Linear(64*4, self.output_steps)

    def forward(self, img):
        # img: N, det, step, channel(NHWC), transform to NCHW for conv
        img = img.permute(0, 3, 1, 2)
        batch_ = img.shape[0]
        img = self.conv(img).permute(0, 2, 1, 3)
        img = img.reshape(batch_, self.num_detector, -1)
        img = self.fc(img)  # flatten to vectors
        return img


if __name__ == '__main__':
    torch.manual_seed(7)

    num_timesteps_input = 12
    num_timesteps_output = 3
    FORECAST_GAP = 2

    lr = 3e-4
    epochs = 1000
    batch_size = 32
    patience = 10

    data_per_day = 192
    sample_per_day = data_per_day - num_timesteps_input - (FORECAST_GAP + 1) * num_timesteps_output + 1

    A, means, stds, training_input, training_target, valid_input, valid_target, test_input, test_target = \
        load_pems_spd_data(input_steps=num_timesteps_input, output_steps=num_timesteps_output,
                           FORECAST_GAP=FORECAST_GAP)

    num_detector = valid_target.shape[1]

    net = Classic_CNN(num_timesteps_input=num_timesteps_input, num_timesteps_output=num_timesteps_output).to(
        device=device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_criterion = nn.MSELoss()

    valid_input = valid_input.to(device=device)
    valid_target = valid_target.to(device=device)

    training_losses, validation_losses, validation_maes, validation_mapes = [], [], [], []

    fast_early_stopping = EarlyStopping(patience=patience, verbose=False)
    for epoch in range(epochs):
        gc.collect()
        torch.cuda.empty_cache()  # at the beginning of each train, clear the unused cache

        train_start = time.time()
        loss = train_epoch(net, optimizer, training_input, training_target, batch_size=batch_size)
        training_losses.append(loss)
        train_time = time.time() - train_start

        # Run validation
        test_batchsize = sample_per_day  # 注意一定要保证验证集和测试集的总数据量都可以整除每天的样本量！！！
        valid_start = time.time()
        with torch.no_grad():
            net.eval()

            # validate set is too large to be passed to cuda all at once
            val_loss, mae, mape = 0.0, 0.0, 0.0
            for i in range(0, valid_input.shape[0], test_batchsize):
                valid_X_batch = valid_input[i:i + test_batchsize]
                valid_y_batch = valid_target[i:i + test_batchsize]
                out = net(valid_X_batch)
                val_loss += loss_criterion(out, valid_y_batch).to(device="cpu").item()

                # inverse-normalization
                out_unnormalized = out.detach().cpu().numpy() * stds[0] + means[0]
                target_unnormalized = valid_y_batch.detach().cpu().numpy() * stds[0] + means[0]
                mae += np.mean(np.absolute(out_unnormalized - target_unnormalized))

                # avoid divide by 0
                target_unnormalized_nonzero = np.copy(target_unnormalized)
                target_unnormalized_nonzero[target_unnormalized_nonzero[:] == 0] = means[0]
                mape += np.mean(
                    np.absolute(target_unnormalized - out_unnormalized) / target_unnormalized_nonzero)

        val_loss /= (valid_input.shape[0] // test_batchsize)
        mae /= (valid_input.shape[0] // test_batchsize)
        mape /= (valid_input.shape[0] // test_batchsize)

        validation_losses.append(val_loss)
        validation_maes.append(mae)
        validation_mapes.append(mape * 100)  # percentage form

        print("epoch %d Train loss: %.6f Valid loss: %.6f mae: %.6f mape: %.6f%% train time %.4f s valid time %.4f s"
              % (epoch + 1, training_losses[-1], validation_losses[-1], validation_maes[-1], validation_mapes[-1],
                 train_time, time.time() - valid_start))

        # fast early stopping criteria
        fast_early_stopping(mae, net)
        # 若满足 early stopping 要求
        if fast_early_stopping.early_stop:
            patience = fast_early_stopping.patience
            # read back best model
            net.load_state_dict(torch.load('LOSS_ANALYSIS/checkpoint.pt'))
            print("Early stopping")
            train_epochs = epoch
            # read back metrics of best model
            train_loss = training_losses[-patience - 1]
            valid_loss = validation_losses[-patience - 1]
            valid_mae = validation_maes[-patience - 1]
            valid_mape = validation_mapes[-patience - 1]
            break

    test_loss, test_mae, test_mape = test_evaluate(net, test_input, test_target)

    if fast_early_stopping.early_stop:
        print("final valid loss %.6f, maes %.6f, mapes %.6f test loss %.6f, mae %.6f, mape %.6f"
              % (valid_loss, valid_mae, valid_mape, test_loss, test_mae, test_mape))
