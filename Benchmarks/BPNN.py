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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(net, optimizer, training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    #  permutation的作用是手动打乱所有样本，也就是一个简易的random sampler
    permutation = torch.randperm(training_input.shape[0])

    loss_criterion = nn.MSELoss()

    epoch_training_losses = []
    #  没有使用DataLoader的sampler，而是直接在数据集上按顺序采样
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        out = net(X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())

    #  average on all batches
    return sum(epoch_training_losses) / len(epoch_training_losses)


def test_evaluate(net, test_input, test_target):
    net.eval()
    test_batchsize = sample_per_day

    test_input = test_input.to(device=device)
    test_target = test_target.to(device=device)

    loss_criterion = nn.MSELoss()

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


class BPNN(nn.Module):
    def __init__(self, num_timesteps_input, num_timesteps_output, num_detector):
        super(BPNN, self).__init__()
        self.num_channel = 2
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.num_nodes = num_detector
        self.hidden_size = 1000
        self.bpnn = nn.Sequential(
            nn.Linear(num_timesteps_input * self.num_nodes * self.num_channel, self.hidden_size),  # *2: 2 channels
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_timesteps_output * self.num_nodes)
        )

    def forward(self, img):
        batch_ = img.shape[0]
        img = img.reshape(batch_, -1)  # flatten to vectors
        output = self.bpnn(img)
        output = output.view(batch_, self.num_nodes, self.num_timesteps_output)
        return output


if __name__ == '__main__':
    torch.manual_seed(7)

    num_timesteps_input = 12
    num_timesteps_output = 3
    FORECAST_GAP = 2

    lr = 1e-4
    epochs = 1000
    batch_size = 512
    patience = 10

    data_per_day = 192
    sample_per_day = data_per_day - num_timesteps_input - (FORECAST_GAP + 1) * num_timesteps_output + 1

    A, means, stds, training_input, training_target, valid_input, valid_target, test_input, test_target = \
        load_pems_spd_data(input_steps=num_timesteps_input, output_steps=num_timesteps_output,
                           FORECAST_GAP=FORECAST_GAP)

    num_detector = valid_target.shape[1]

    net = BPNN().to(device=device)

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

