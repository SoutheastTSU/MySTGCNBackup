import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import torch.nn as nn
from ST_Conv import STConv
from Benchmarks.BPNN import BPNN
from Benchmarks.CNN import Classic_CNN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, EarlyStopping, ChebPolynomial
from data.PeMS import load_pems_spd_data
from PipeLine.main import OutputData, MyModel

warnings.filterwarnings('error')  # catch warnings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EvalModel(MyModel):
    def init_and_load_model_param(self, param_dict_path: str):
        self.init_model()
        self.net = STConv(training_input.shape[1],
                          training_input.shape[3],
                          self.num_timesteps_input,
                          self.num_timesteps_output, self.Ks, self.Kt).to(device=device)
        self.net.load_state_dict(torch.load(param_dict_path))  # load the path using torch.load, then load the params

    def forecast_on_data(self, eval_input: torch.Tensor, eval_target: torch.Tensor) -> np.ndarray:
        """
            recommend input whole test set data, or at least data of a single day(shape[0] == self.test_batchsize)
            eval_input and eval_target must have same batches and num_nodes
            return the forecast result of eval_input(np.ndarray)
        """
        # eval data should in shape(batch, nodes, time)
        self.net.eval()
        eval_input = eval_input.to(device)
        eval_target = eval_target.to(device)
        mse, mae, mape = 0.0, 0.0, 0.0
        with torch.no_grad():  # memory will explode without this 不加这句内存直接爆
            # forecast small batches to avoid running out of memory
            for i in range(eval_input.shape[0] // self.test_batchsize):
                gc.collect()
                torch.cuda.empty_cache()  # at the beginning of each train, clear the unused cache

                batch_slice = slice(i * self.test_batchsize, (i + 1) * self.test_batchsize)
                out = self.net(eval_input[batch_slice])
                gt_y = eval_target[batch_slice]

                if i == 0:
                    forecast_result = out
                else:
                    forecast_result = torch.cat((forecast_result, out), 0)

                mse += ((out - gt_y) ** 2).mean(axis=0).detach().cpu().numpy() * self.stds[0]**2

                # inverse-normalization
                out_unnormalized = out.detach().cpu().numpy() * self.stds[0] + self.means[0]
                target_unnormalized = gt_y.detach().cpu().numpy() * self.stds[0] + self.means[0]
                mae += np.mean(np.absolute(out_unnormalized - target_unnormalized), axis=0)

                # avoid divide by 0
                target_unnormalized_nonzero = np.copy(target_unnormalized)
                target_unnormalized_nonzero[target_unnormalized_nonzero[:] == 0] = self.means[0]
                mape += np.mean(np.absolute(target_unnormalized - out_unnormalized) / target_unnormalized_nonzero,
                                axis=0)

        result = forecast_result.cpu().numpy() * self.stds[0] + self.means[0]
        # eval_target = eval_target.numpy() * self.stds[0] + self.means[0]

        mse /= (eval_input.shape[0] // self.test_batchsize)
        mae /= (eval_input.shape[0] // self.test_batchsize)
        mape /= (eval_input.shape[0] // self.test_batchsize)

        # test_mse = np.mean(mse)
        # test_mae = np.mean(mae)
        # test_mape = np.mean(mape)

        mean_time_mae = np.mean(mae, axis=0)
        mean_detector_mae = np.mean(mae, axis=1)
        mean_time_mse = np.mean(mse * stds ** 2, axis=0)
        mean_detector_mse = np.mean(mse, axis=1)
        # mean_time_mape = np.mean(mape, axis=0)
        # mean_detector_mape = np.mean(mape, axis=1)

        # plot
        # plt.figure()
        # plt.plot(mean_time_mae)
        # plt.ylabel('MAEs', fontsize=18)
        # plt.xlabel('forecast steps', fontsize=18)

        # plt.figure()
        # plt.plot(mean_detector_mae)
        # plt.axhline(y=mean_detector_mae.mean(), ls='--', color='r')
        # plt.ylabel('MAEs', fontsize=18)
        # plt.xlabel('detector', fontsize=18)
        #
        # plt.figure()
        # plt.plot(mean_detector_mse)
        # plt.axhline(y=mean_detector_mse.mean(), ls='--', color='r')
        # plt.ylabel('MSEs', fontsize=18)
        # plt.xlabel('detector', fontsize=18)

        # fig, ax0 = plt.subplots()
        # c = ax0.pcolor(loss, cmap='Greys')  # cmap: name of chosen colormap, see
        # # https://matplotlib.org/3.5.3/tutorials/colors/colormaps.html
        # ax0.set_title(f'Speed Prediction MSE', fontsize=20)
        # plt.ylabel('detectors', fontsize=18)
        # plt.xlabel('forecast steps', fontsize=18)

        # fig, ax0 = plt.subplots()
        # c = ax0.pcolor(mape, cmap='Greys')  # cmap: name of chosen colormap, see
        # # https://matplotlib.org/3.5.3/tutorials/colors/colormaps.html
        # ax0.set_title(f'Speed Prediction MAPE', fontsize=20)
        # plt.ylabel('detectors', fontsize=18)
        # plt.xlabel('forecast steps', fontsize=18)
        # # ticks = [0, 24, 48, 72, 96, 120, 144, 168, 192]
        # # labels = ['5:00', '7:00', '9:00', '11:00', '13:00', '15:00', '17:00', '19:00', '21:00']
        # # plt.yticks(ticks=ticks, labels=labels, fontsize=16)
        # # plt.xticks(fontsize=18)
        # fig.tight_layout()

        plt.show()

        """
            这一段是画直方图统计预测结果在各个速度区间的分布
            plot histogram showing distribution of forecast speed results in different speed ranges
        """
        # # plot histogram showing distributions of predicted data
        # forecast_result = result.reshape(-1, 3)
        # eval_target = eval_target.reshape(-1, 3)
        # plt.figure()
        # y_min = forecast_result.min()
        # y_max = forecast_result.max()
        # plt.hist(forecast_result, bins=[y_min, 0, 20, 60, y_max])
        # plt.figure()
        # plt.hist(eval_target, bins=[y_min, 0, 20, 60, y_max])
        # plt.show()

        return result

    def forecast_one_day_one_det(self, day_idx: int, det_idx: int, eval_data: torch.Tensor, eval_target: torch.Tensor):
        # # # # plot comparison of forecast data and ground truth data
        # results
        result = self.forecast_on_data(eval_data, eval_target)
        batch_slice = slice(self.test_batchsize * day_idx, (day_idx + 1) * self.test_batchsize)
        result = result[batch_slice, det_idx, :]

        # ground truth data
        tmp_gt = eval_target[batch_slice, det_idx, 0]
        # [[0,3,6],[1,4,7],...,[i,i+3,i+6]], so ground_truth = first column + last gap*width elements of last column
        gap, width = 3, eval_target.shape[-1] - 1
        tmp_gt = np.append(tmp_gt, eval_target[-gap * width:, det_idx, -1]) * self.stds[0] + self.means[0]
        plt.plot(tmp_gt, 'b-', label='ground truth')  # plot gt data in blue line

        for i in range(self.test_batchsize):
            x_label = np.array([i + gap * j for j in range(width + 1)])
            plt.plot(x_label, result[i], 'r-')
        plt.legend()
        plt.show()

    def compare_multistep(self, day_idx: int, det_idx: int, eval_data: torch.Tensor, eval_target: torch.Tensor):
        # # # # plot comparison of forecast data and ground truth data
        # results
        result = self.forecast_on_data(eval_data, eval_target)
        batch_slice = slice(self.test_batchsize * day_idx, (day_idx + 1) * self.test_batchsize)
        result = result[batch_slice, det_idx, :]

        # ground truth data
        tmp_gt0 = eval_target[batch_slice, det_idx, 0]
        # [[0,3,6],[1,4,7],...,[i,i+3,i+6]], so ground_truth = first column + last gap*width elements of last column
        gap, width = 3, eval_target.shape[-1] - 1
        tmp_gt = np.append(tmp_gt0, eval_target[-gap * width:, det_idx, -1]) * self.stds[0] + self.means[0]
        plt.plot(tmp_gt, 'b-', label='ground truth')  # plot gt data in blue line

        # No prediction
        lag = 9
        no_prediction = np.append(np.zeros((lag, 1)), tmp_gt0[:-lag]) * self.stds[0] + self.means[0]
        plt.plot(no_prediction, 'r--', label='no prediction')

        # 画出多步预测的结果（每一种预测步的结果画成一条线）
        for i in range(width, width+1):
            x_label = np.array([j + i*gap for j in range(self.test_batchsize)])
            plt.plot(x_label, result[:, i], label=str((1+i)*15)+' min')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # torch.manual_seed(7)

    num_timesteps_input = 12
    num_timesteps_output = 3
    FORECAST_GAP = 2

    # means & stds: ndarray(1,)     inputs and targets: tensor(batch, nodes, steps, (channel))
    _, means, stds, training_input, training_target, valid_input, valid_target, test_input, test_target = \
        load_pems_spd_data(input_steps=num_timesteps_input, output_steps=num_timesteps_output,
                           FORECAST_GAP=FORECAST_GAP, with_history=True)

    data_per_day = 192
    sample_per_day = data_per_day - num_timesteps_input - (FORECAST_GAP + 1) * num_timesteps_output + 1

    input_dict = {'num_nodes': training_input.shape[1],
                  'num_features': training_input.shape[3],
                  'num_timesteps_input': num_timesteps_input,
                  'num_timesteps_output': num_timesteps_output,
                  'sample_per_day': sample_per_day,
                  'flag_plot_loss': True,
                  'means': means, 'stds': stds,
                  'training_input': training_input, 'training_target': training_target,
                  'valid_input': valid_input, 'valid_target': valid_target,
                  'test_input': test_input, 'test_target': test_target
                  }

    model_dict = {'class_name': 'STConv', 'Ks': 3, 'Kt': 3, 'lr': 1e-3, 'epochs': 1000, 'batch_size': 32,
                  'patience': 10}

    model = EvalModel(**input_dict, **model_dict)
    param_dict = r'F:\Graduate\AverageSpeedPrediction\STGCN_PyTorch\STGCN-PyTorch-master\PipeLine\LOSS_ANALYSIS\STConvWithHistory_checkpoint.pt'
    model.init_and_load_model_param(param_dict)
    # forecast_result = model.forecast_on_data(test_input, test_target)
    # model.forecast_one_day_one_det(eval_data=test_input, eval_target=test_target, day_idx=1, det_idx=50)
    model.compare_multistep(eval_data=test_input, eval_target=test_target, day_idx=8, det_idx=1)
    print('break')
