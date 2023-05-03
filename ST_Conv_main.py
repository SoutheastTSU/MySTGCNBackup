import os
import argparse
import pickle as pk
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import torch.nn as nn
from ST_Conv import STConv
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, EarlyStopping, ChebPolynomial
from data.PeMS import load_pems_spd_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_validate(valid_input, valid_target):
    net = STConv(valid_input.shape[1],
                 training_input.shape[3],
                 num_timesteps_input,
                 num_timesteps_output, Ks, Kt).to(device=device)
    # num_nodes, num_features, num_timesteps_input, num_timesteps_output

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_criterion = nn.MSELoss()

    valid_input = valid_input.to(device=device)
    valid_target = valid_target.to(device=device)

    training_losses, validation_losses, validation_maes, validation_mapes = [], [], [], []

    patience = 10
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

    return training_losses[0:-patience], validation_losses[0:-patience], validation_maes[0:-patience], \
           validation_mapes[0:-patience], test_loss, test_mae, test_mape


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
            loss += loss_criterion(out, test_y_batch).to(device="cpu").item()

            # inverse-normalization
            out_unnormalized = out.detach().cpu().numpy() * stds[0] + means[0]
            target_unnormalized = test_y_batch.detach().cpu().numpy() * stds[0] + means[0]
            mae += np.mean(np.absolute(out_unnormalized - target_unnormalized))

            # avoid divide by 0
            target_unnormalized_nonzero = np.copy(target_unnormalized)
            target_unnormalized_nonzero[target_unnormalized_nonzero[:] == 0] = means[0]
            mape += np.mean(np.absolute(target_unnormalized - out_unnormalized) / target_unnormalized_nonzero)

    loss /= (test_input.shape[0] // test_batchsize)
    mae /= (test_input.shape[0] // test_batchsize)
    mape /= (test_input.shape[0] // test_batchsize)

    return loss, mae, mape


if __name__ == '__main__':
    torch.manual_seed(7)

    num_timesteps_input = 12
    num_timesteps_output = 3
    FORECAST_GAP = 2  # e.g.: data timestep is 5min, but forecast step is 15min, so there is GAP==2 steps for every forecast

    lr = 1e-3
    epochs = 1000
    batch_size = 32

    Ks = 3
    Kt = 3

    data_per_day = 192
    sample_per_day = data_per_day - num_timesteps_input - (FORECAST_GAP + 1) * num_timesteps_output + 1

    _, means, stds, training_input, training_target, valid_input, valid_target, test_input, test_target = \
        load_pems_spd_data(input_steps=num_timesteps_input, output_steps=num_timesteps_output,
                           FORECAST_GAP=FORECAST_GAP)

    # #  A: Adjacent Matrix. here A_Wave == D_wave*(A+I)*D_wave.T
    # A_wave = get_normalized_adj(A)
    # Cheb_Poly = ChebPolynomial(A_wave, Ks).to(device=device)
    # # A_wave = torch.from_numpy(A_wave).to(device=device)


    valid_mae_trace, test_maes, valid_mse_trace, test_mses, valid_mape_trace, test_mapes = [], [], [], [], [], []

    fig, subs = plt.subplots(2, 2)
    subs[0][0].set_title('train_loss')
    subs[0][1].set_title('valid_loss')
    subs[1][0].set_title('valid_mae')
    subs[1][1].set_title('valid_mape')

    training_times = 2
    for training_time in range(training_times):
        # training, validation and testing
        train_losses, valid_losses, valid_maes, valid_mapes, test_mse, test_mae, test_mape = train_and_validate(
            valid_input, valid_target)
        # valid_mse_trace.append(valid_losses)
        # valid_mae_trace.append(valid_maes)
        # valid_mape_trace.append(valid_mapes)
        test_mses.append(test_mse)
        test_maes.append(test_mae)
        test_mapes.append(test_mape)

        subs[0][0].plot(train_losses, label=str(training_times))
        subs[0][1].plot(valid_losses, label=str(training_times))
        subs[1][0].plot(valid_maes, label=str(training_times))
        subs[1][1].plot(valid_mapes, label=str(training_times))

    subs[0][0].legend()
    subs[0][1].legend()
    subs[1][0].legend()
    subs[1][1].legend()

    mse = np.array(test_mses)
    mae = np.array(test_maes)
    mape = np.array(test_mapes)
    mse_mean = mse.mean()
    mae_mean = mae.mean()
    mape_mean = mape.mean()
    mse_std = mse.std()
    mae_std = mae.std()
    mape_std = mape.std()

    # plt.figure(2)
    # plt.plot(batch_size_lst, test_maes, label='test mae')
    # plt.xlabel('batch size')
    # plt.title('test mae')
    # plt.xticks(ticks=batch_size_lst, labels=batch_size_lst)
    # # test_axis.plot(test_mses, label='test mape')
    # # test_axis.plot(test_mses, label='test mse')
    # plt.show()

    # plot
    # plt.figure()
    # plt.plot(train_losses, label="training loss")
    # plt.plot(valid_losses, label="validation loss")
    # plt.title("Loss Curve")
    # plt.legend()
    # plt.figure()
    # plt.plot(valid_maes, label="validation mae")
    # plt.plot(valid_mapes, label="validation mape(%)")
    # plt.title("Validation MAEs and MAPEs")
    # plt.legend()
    # plt.show()

    print('break point')
