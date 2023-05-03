import os
import zipfile
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'LOSS_ANALYSIS/checkpoint.pt')  # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model.pkl')                   # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss


# #  加载数据并且分通道归一化，归一化方式是减去均值再除以标准差
# def load_metr_la_data():
#     if (not os.path.isfile("data/adj_mat.npy")
#             or not os.path.isfile("data/node_values.npy")):
#         with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
#             zip_ref.extractall("data/")
#
#     A = np.load("data/adj_mat.npy")
#     X = np.load("data/node_values.npy").transpose((1, 2, 0))
#     X = X.astype(np.float32)
#
#     # Normalization using Z-score method
#     means = np.mean(X, axis=(0, 2))
#     X = X - means.reshape(1, -1, 1)
#     stds = np.std(X, axis=(0, 2))
#     X = X / stds.reshape(1, -1, 1)
#
#     return A, X, means, stds


#  加载数据并且分通道归一化，归一化方式是减去均值再除以标准差
def load_metr_la_data():
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")
    X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds


#  虽然名字是A_wave，但是实际上对应论文里D_wave^(-0.5)*W_wave*D_wave^(-0.5).T, 其中D_wave::D, W_wave::A
def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    # A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  # equal to 'A + np.eye(A.shape[0])'
    D = np.array(np.sum(A, axis=1), dtype='float32').reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.eye(A.shape[0]) - np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_wave = A_wave.astype('float32')
    return A_wave


def ChebPolynomial(L_wave, K):
    L_wave = torch.from_numpy(L_wave)
    x_0 = torch.eye(L_wave.shape[0])
    x_1 = L_wave
    x_list = [x_0, x_1]
    for k in range(2, K):
        x_list.append(torch.matmul(2 * L_wave, x_list[k - 1]) - x_list[k - 2])
    x = torch.stack(x_list, dim=0)
    return x

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timestep
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))
