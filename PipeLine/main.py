# -*- coding: UTF-8 -*-
# import math
# import torch
# import torch.nn.functional as F
# import os
# import argparse
# import pickle as pk
# import os
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import torch.nn as nn
from ST_Conv import STConv
from Benchmarks.BPNN import BPNN
from Benchmarks.CNN import Classic_CNN, ImprovedFC_CNN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, EarlyStopping, ChebPolynomial
from data.PeMS import load_pems_spd_data, load_mobile_century_spd_data

warnings.filterwarnings('error')  # catch warnings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OutputData:
    def __init__(self):
        self.loss_curve = []  # average train loss in each epoch
        self.iter_loss = []  # train loss of each iteration(each batch)
        self.valid_loss_curve = []  # average valid loss in each epoch
        self.valid_mae_curve = []
        self.valid_mape_curve = []

        # shape == (nodes, timestep), avg error on each node at each timestep
        self.valid_mse_plain, self.valid_mae_plain, self.valid_mape_plain = None, None, None
        self.test_mse_plain, self.test_mae_plain, self.test_mape_plain = None, None, None

        self.valid_lst = np.zeros(3)  # in order: mse, mae, mape
        self.test_lst = np.zeros(3)  # 3: number of metrics
        self.forecast_valid = None  # 3 list: mse, mae, mape
        self.forecast_test = None

    def plot_loss(self):
        fig1 = plt.figure()
        self.iter_loss = np.array(self.iter_loss).reshape(-1)
        plt.plot(self.iter_loss, label='loss on each iter')
        plt.title("train loss on each iter")

        fig2 = plt.figure()
        plt.plot(self.loss_curve, label='loss')
        plt.plot(self.valid_loss_curve, label='valid loss')
        plt.plot(self.valid_mae_curve, label='valid mae')
        plt.plot(self.valid_mape_curve, label='valid mape')
        plt.title("train & valid loss")
        plt.legend()
        plt.show()
        plt.close(fig1)
        plt.close(fig2)

    def plot_output_data(self, ax_valid, ax_test):
        ax_valid.plot(self.forecast_valid[1])  # plot mae only
        ax_test.plot(self.forecast_test[1])

    def print_one_train_result(self):
        # forecast performance on each forecast step
        if self.valid_mse_plain is not None:
            self.forecast_valid = np.array([np.mean(self.valid_mse_plain, axis=0),
                                            np.mean(self.valid_mae_plain, axis=0),
                                            np.mean(self.valid_mape_plain, axis=0)])

        if self.test_mse_plain is not None:
            self.forecast_test = np.array([np.mean(self.test_mse_plain, axis=0),
                                           np.mean(self.test_mae_plain, axis=0),
                                           np.mean(self.test_mape_plain, axis=0)])

        print("final valid loss %.6f, maes %.6f, mapes %.6f test loss %.6f, mae %.6f, mape %.6f"
              % (self.valid_lst[0], self.valid_lst[1], self.valid_lst[2],
                 self.test_lst[0], self.test_lst[1], self.test_lst[2]))


class MyModel:
    def __init__(self, class_name, means, stds, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, sample_per_day, Ks, Kt, lr, epochs, batch_size, patience, flag_plot_loss,
                 training_input, training_target, valid_input, valid_target, test_input, test_target):
        self.loss_criterion = None
        self.net = None
        self.optimizer = None
        self.class_name = class_name
        self.input_channel = training_input.shape[-1]
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.Ks = Ks
        self.Kt = Kt
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.test_batchsize = sample_per_day  # 注意一定要保证验证集和测试集的总数据量都可以整除每天的样本量！！！
        self.flag_plot_loss_curve = flag_plot_loss
        self.means, self.stds = means, stds
        self.training_input, self.training_target = training_input, training_target
        self.valid_input, self.valid_target = valid_input, valid_target
        self.test_input, self.test_target = test_input, test_target

        # data to be saved
        self.OutputData_list = []

    def init_model(self):
        # instantiate class: class_name
        if self.class_name == 'STConv':
            self.net = STConv(self.training_input.shape[1],
                              self.training_input.shape[3],
                              self.num_timesteps_input,
                              self.num_timesteps_output, self.Ks, self.Kt).to(device=device)
        elif self.class_name == 'BPNN':
            self.net = BPNN(self.num_timesteps_input, self.num_timesteps_output, self.num_nodes).to(device=device)
        elif self.class_name == 'CNN':
            self.net = Classic_CNN(self.num_timesteps_input, self.num_timesteps_output, self.num_nodes).to(
                device=device)
        elif self.class_name == 'ImprovedCNN':
            self.net = ImprovedFC_CNN(self.num_timesteps_input, self.num_timesteps_output, self.num_nodes, self.input_channel).to(device=device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_criterion = nn.MSELoss()

    def train_and_validate(self):
        new_output = OutputData()

        # the scope of early_stopping: within train_and_validate() function
        fast_early_stopping = EarlyStopping(patience=self.patience, verbose=False)
        for epoch in range(self.epochs):
            gc.collect()
            torch.cuda.empty_cache()  # at the beginning of each train, clear the unused cache

            train_start = time.time()

            loss_on_batch, loss = self.train_epoch()  # loss on each single batch and average loss on all batches
            new_output.iter_loss.append(loss_on_batch)
            new_output.loss_curve.append(loss)

            train_time = time.time() - train_start

            # Run validation
            valid_start = time.time()

            val_loss, mae, mape = self.validate_epoch()  # output shape(nodes, timestep)

            mean_mae = np.mean(mae)
            mean_valid_loss = np.mean(val_loss)
            new_output.valid_loss_curve.append(np.mean(val_loss))
            new_output.valid_mae_curve.append(mean_mae)
            new_output.valid_mape_curve.append(np.mean(mape))

            print(
                "epoch %d Train loss: %.6f Valid loss: %.6f mae: %.6f mape: %.6f train time %.4f s valid time %.4f s"
                % (epoch + 1, new_output.loss_curve[-1], new_output.valid_loss_curve[-1],
                   new_output.valid_mae_curve[-1], new_output.valid_mape_curve[-1],
                   train_time, time.time() - valid_start))

            # fast early stopping criteria
            # fast_early_stopping(mean_mae, self.net)
            fast_early_stopping(mean_valid_loss, self.net)
            if fast_early_stopping.early_stop:
                early_stop_patience = fast_early_stopping.patience
                # read back best model
                self.net.load_state_dict(torch.load('LOSS_ANALYSIS/checkpoint.pt'))

                mse_plain, new_output.valid_mae_plain, mape_plain = self.validate_epoch()
                new_output.valid_mse_plain = mse_plain * self.stds[0] ** 2
                new_output.valid_mape_plain = mape_plain * 100  # percentage form

                print("Early stopping")
                # train_epochs = epoch
                # read back metrics of best model
                new_output.valid_lst[0] = new_output.valid_loss_curve[
                                              -early_stop_patience - 1] * self.stds[0] ** 2  # inverse-normalization
                new_output.valid_lst[1] = new_output.valid_mae_curve[-early_stop_patience - 1]
                new_output.valid_lst[2] = new_output.valid_mape_curve[-early_stop_patience - 1] * 100
                break

        test_mse_plain, new_output.test_mae_plain, test_mape_plain = self.test_evaluate(self.test_input, self.test_target)
        new_output.test_mse_plain = test_mse_plain * self.stds[0] ** 2  # inverse normalization
        new_output.test_mape_plain = test_mape_plain * 100  # percentage form

        new_output.test_lst[0] = np.mean(test_mse_plain) * self.stds[0] ** 2
        new_output.test_lst[1] = np.mean(new_output.test_mae_plain)
        new_output.test_lst[2] = np.mean(test_mape_plain)

        new_output.print_one_train_result()
        self.OutputData_list.append(new_output)

    def train_epoch(self):
        """
        Trains one epoch with the given data.
        :param training_input: Training inputs of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param training_target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :return: Average loss for this epoch.
        """
        self.net.train()

        #  permutation的作用是手动打乱所有样本，也就是一个简易的random sampler
        permutation = torch.randperm(self.training_input.shape[0])

        epoch_training_losses = []
        #  没有使用DataLoader的sampler，而是直接在数据集上按顺序采样
        for i in range(0, self.training_input.shape[0], self.batch_size):
            self.net.train()
            self.optimizer.zero_grad()

            indices = permutation[i:i + self.batch_size]
            X_batch, y_batch = self.training_input[indices], self.training_target[indices]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)

            out = self.net(X_batch)
            loss = self.loss_criterion(out, y_batch)
            loss.backward()
            self.optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())

        #  average on all batches
        return epoch_training_losses, sum(epoch_training_losses) / len(epoch_training_losses)

    def validate_epoch(self):
        with torch.no_grad():
            self.net.eval()

            # validate set is too large to be passed to cuda all at once
            val_loss, mae, mape = 0.0, 0.0, 0.0
            for i in range(0, self.valid_input.shape[0], self.test_batchsize):
                valid_X_batch = self.valid_input[i:i + self.test_batchsize].to(device)
                valid_y_batch = self.valid_target[i:i + self.test_batchsize].to(device)
                out = self.net(valid_X_batch)
                val_loss += ((out - valid_y_batch) ** 2).mean(axis=0).detach().cpu().numpy()

                # inverse-normalization
                out_unnormalized = out.detach().cpu().numpy() * self.stds[0] + self.means[0]
                target_unnormalized = valid_y_batch.detach().cpu().numpy() * self.stds[0] + self.means[0]
                mae += np.mean(np.absolute(out_unnormalized - target_unnormalized), axis=0)

                # avoid divide by 0
                target_unnormalized_nonzero = np.copy(target_unnormalized)
                target_unnormalized_nonzero[target_unnormalized_nonzero[:] == 0] = self.means[0]
                mape += np.mean(np.absolute(target_unnormalized - out_unnormalized) / target_unnormalized_nonzero,
                                axis=0)
            batch_num = (self.valid_input.shape[0] // self.test_batchsize)
            val_loss, mae, mape = map(lambda x: x / batch_num, [val_loss, mae, mape])

            self.net.train()

        return val_loss, mae, mape  # mse normalized, mae unnormalized, mape is raw

    def test_evaluate(self, test_input, test_target):
        self.net.eval()
        test_batchsize = sample_per_day

        test_input = test_input.to(device=device)
        test_target = test_target.to(device=device)

        # validate set is too large to be passed to cuda all at once
        mse, mae, mape = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i in range(0, test_input.shape[0], test_batchsize):
                test_X_batch = test_input[i:i + test_batchsize]
                test_y_batch = test_target[i:i + test_batchsize]
                out = self.net(test_X_batch)
                mse += ((out - test_y_batch) ** 2).mean(axis=0).detach().cpu().numpy()

                # inverse-normalization
                out_unnormalized = out.detach().cpu().numpy() * self.stds[0] + self.means[0]
                target_unnormalized = test_y_batch.detach().cpu().numpy() * self.stds[0] + self.means[0]
                mae += np.mean(np.absolute(out_unnormalized - target_unnormalized), axis=0)

                # avoid divide by 0
                target_unnormalized_nonzero = np.copy(target_unnormalized)
                target_unnormalized_nonzero[target_unnormalized_nonzero[:] == 0] = self.means[0]
                mape += np.mean(np.absolute(target_unnormalized - out_unnormalized) / target_unnormalized_nonzero,
                                axis=0)

        mse /= (test_input.shape[0] // test_batchsize)
        mae /= (test_input.shape[0] // test_batchsize)
        mape /= (test_input.shape[0] // test_batchsize)

        return mse, mae, mape  # mse normalized, mae unnormalized, mape is raw

    def result(self):
        avg_forecast_valid = np.zeros((3, num_timesteps_output))  # mse, mae, mape
        avg_forecast_test = np.zeros((3, num_timesteps_output))
        std_forecast_valid = np.zeros((3, num_timesteps_output))  # stds = E(X^2) - E(X)^2
        std_forecast_test = np.zeros((3, num_timesteps_output))

        for out in self.OutputData_list:
            avg_forecast_valid += out.forecast_valid
            avg_forecast_test += out.forecast_test
            std_forecast_valid += out.forecast_valid ** 2
            std_forecast_test += out.forecast_test ** 2
            if self.flag_plot_loss_curve:
                out.plot_loss()

        # average
        length = len(self.OutputData_list)
        [avg_forecast_valid, avg_forecast_test, std_forecast_valid, std_forecast_test] = \
            map(lambda x: x / length, [avg_forecast_valid, avg_forecast_test, std_forecast_valid, std_forecast_test])
        std_forecast_valid = std_forecast_valid - avg_forecast_valid ** 2
        std_forecast_test = std_forecast_test - avg_forecast_test ** 2

        _, _, _, _ = map(lambda x, y: print(f'{y} mse on each step: {x[0, :]}, mae: {x[1, :]}, mape:{x[2, :]}'),
                         [avg_forecast_valid, avg_forecast_test, std_forecast_valid, std_forecast_test],
                         ['avg valid', 'avg test', 'stds valid', 'stds test'])


if __name__ == '__main__':
    # torch.manual_seed(7)

    num_timesteps_input = 12
    num_timesteps_output = 3
    FORECAST_GAP = 2

    # _, means, stds, training_input, training_target, valid_input, valid_target, test_input, test_target = \
    #     load_pems_spd_data(input_steps=num_timesteps_input, output_steps=num_timesteps_output,
    #                        FORECAST_GAP=FORECAST_GAP, with_history=True)
    _, means, stds, training_input, training_target, valid_input, valid_target, test_input, test_target = \
        load_mobile_century_spd_data(input_steps=num_timesteps_input, output_steps=num_timesteps_output,
                           FORECAST_GAP=FORECAST_GAP, with_history=True)

    data_per_day = 288
    sample_per_day = data_per_day - num_timesteps_input - (FORECAST_GAP + 1) * num_timesteps_output + 1

    # input_dict = {'num_nodes': training_input.shape[1],
    #               'num_features': training_input.shape[3],
    #               'num_timesteps_input': num_timesteps_input,
    #               'num_timesteps_output': num_timesteps_output,
    #               'sample_per_day': sample_per_day,
    #               'flag_plot_loss': True,
    #               'means': means, 'stds': stds,
    #               'training_input': training_input, 'training_target': training_target,
    #               'valid_input': valid_input, 'valid_target': valid_target,
    #               'test_input': test_input, 'test_target': test_target
    #               }
    #
    # model_dict = {'class_name': 'STConv', 'Ks': 3, 'Kt': 3, 'lr': 1e-3, 'epochs': 1000, 'batch_size': 64,
    #               'patience': 5}
    # model = MyModel(**input_dict, **model_dict)

    # CNN_input_dict = {'class_name': 'CNN',
    #               'num_nodes': training_input.shape[1],
    #               'num_features': training_input.shape[3],
    #               'num_timesteps_input': num_timesteps_input,
    #               'num_timesteps_output': num_timesteps_output,
    #               'sample_per_day': sample_per_day,
    #               'Ks': 3, 'Kt': 3,
    #               'lr': 1e-4, 'epochs': 1000,
    #               'batch_size': 64,
    #               'patience': 5,
    #               'flag_plot_loss': False,
    #               'means': means, 'stds': stds,
    #               'training_input': training_input, 'training_target': training_target,
    #               'valid_input': valid_input, 'valid_target': valid_target,
    #               'test_input': test_input, 'test_target': test_target}
    # model = MyModel(**CNN_input_dict)

    ImprovedCNN_input_dict = {'class_name': 'ImprovedCNN',
                      'num_nodes': training_input.shape[1],
                      'num_features': training_input.shape[3],
                      'num_timesteps_input': num_timesteps_input,
                      'num_timesteps_output': num_timesteps_output,
                      'sample_per_day': sample_per_day,
                      'Ks': 3, 'Kt': 3,
                      'lr': 1e-4, 'epochs': 1000,
                      'batch_size': 64,
                      'patience': 5,
                      'flag_plot_loss': False,
                      'means': means, 'stds': stds,
                      'training_input': training_input, 'training_target': training_target,
                      'valid_input': valid_input, 'valid_target': valid_target,
                      'test_input': test_input, 'test_target': test_target}
    model = MyModel(**ImprovedCNN_input_dict)

    # BPNN_input_dict = {'class_name': 'BPNN',
    #                   'num_nodes': training_input.shape[1],
    #                   'num_features': training_input.shape[3],
    #                   'num_timesteps_input': num_timesteps_input,
    #                   'num_timesteps_output': num_timesteps_output,
    #                   'sample_per_day': sample_per_day,
    #                   'Ks': 3, 'Kt': 3,
    #                   'lr': 1e-4, 'epochs': 1000,
    #                   'batch_size': 64,
    #                   'patience': 5,
    #                   'flag_plot_loss': False,
    #                   'means': means, 'stds': stds,
    #                   'training_input': training_input, 'training_target': training_target,
    #                   'valid_input': valid_input, 'valid_target': valid_target,
    #                   'test_input': test_input, 'test_target': test_target}
    # model = MyModel(**BPNN_input_dict)

    # init_model() and then train_and_validate(). 在这两个函数外套上各种循环，以重复训练多次
    # for i in range(10):
    model.init_model()
    model.train_and_validate()

    model.result()

    print('set break point')
