import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""original TimeBlock"""


class Original_TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))  # 1-D casual conv, kernel:(1, kernel_size)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))  # really??? 这里没写错???
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))  # 1-D casual conv, kernel:(1, kernel_size)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        x_p = self.conv1(X)
        x_q = self.conv2(X)

        if X.shape[1] < self.out_channels:  # use zeros to compensate
            batch_size, in_channels, num_nodes, num_timesteps = X.shape
            X = torch.cat(
                [X, torch.zeros([batch_size, self.out_channels - in_channels, num_nodes, num_timesteps]).to(X)], dim=1)

        # 这里修改了时间卷积以及Res连接的方式
        out = torch.mul((X[:, :, :, :-(self.kernel_size - 1)] + x_p), torch.sigmoid(x_q))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, in_timesteps, Ks):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.Ks = Ks
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        # ChebGraphConv Kernel
        self.Theta1 = nn.Parameter(torch.FloatTensor(Ks, out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, cheb):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        cheb: (Ks, Nodes, Nodes)
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # X: batch, num_nodes, timestep, channel
        t0 = self.temporal1(X).permute(0, 3, 1, 2)  # t0 -> (batch, channel, num_nodes, timestep)

        if self.Ks - 1 < 0:
            raise ValueError(
                f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')

        # ChebGraphConv
        Cheb_x = torch.einsum('kmn,bcnt->bckmt', cheb, t0)  # m==n==nodes; b:batch; c:channel; k:kernel; t:timestep;
        t2 = torch.einsum('bckmt,kco->bmto', Cheb_x, self.Theta1)  # o:out_channel;
        t3 = F.relu(t2)  # t3: batch, nodes, timestep, channel
        # x_0 = t0
        # x_1 = torch.einsum('ih,nchw->nciw', A_hat, t0)  # i==h==Nodes, n:batch_size, c:channel, w:timestep
        # x_list = [x_0, x_1]
        # for k in range(2, self.Ks):
        #     x_list.append(torch.einsum('ih,nchw->nciw', 2 * A_hat, x_list[k - 1]) - x_list[k - 2])
        #
        # x = torch.stack(x_list, dim=2)
        #
        # # j: out_channel; c:in_channel; k: kernel_size; i:num_nodes; w: num_timesteps
        # t2 = F.relu(torch.einsum('nckiw,kcj->niwj', x_SAtt, self.Theta1))
        t4 = self.temporal2(t3)
        return self.batch_norm(t4)


"""original STGCNBlock"""


class Origin_STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))  # Graph Conv Kernel
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])  # i,j:=num_nodes;
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))  # 这里只进行了一次图卷积？？？但是论文里面K=3，应该连续进行三次才对啊？
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, Ks, Kt, ChebPoly):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.spatial_channel = 16
        self.temporal_channel = 64
        self.Cheb = ChebPoly
        self.Kt = Kt  # Kernel size of temporal conv
        self.Ks = Ks
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=self.temporal_channel,
                                 spatial_channels=self.spatial_channel, num_nodes=num_nodes,
                                 in_timesteps=num_timesteps_input, Ks=Ks)
        self.block2 = STGCNBlock(in_channels=self.temporal_channel, out_channels=self.temporal_channel,
                                 spatial_channels=self.spatial_channel, num_nodes=num_nodes,
                                 in_timesteps=num_timesteps_input - 2 * (Kt - 1), Ks=Ks)
        self.last_temporal = TimeBlock(in_channels=self.temporal_channel, out_channels=self.temporal_channel)
        self.fully1 = nn.Linear((num_timesteps_input - (Kt - 1) * 5) * self.temporal_channel, num_timesteps_output)
        # self.fully2 = nn.Linear(128, num_timesteps_output)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, self.Cheb)
        out2 = self.block2(out1, self.Cheb)
        out3 = self.last_temporal(out2)
        out4 = self.fully1(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # out5 = self.fully2(F.relu(out4))
        return out4
