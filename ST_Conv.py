import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class STConvBlock(nn.Module):
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
        super(STConvBlock, self).__init__()
        self.Ks = Ks
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)

        # 1-D spatial conv
        if self.Ks % 2 != 1:  # self.Ks should be odd number
            raise ValueError(
                f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')

        # different padding size on height and width: padding=(height_padding, width_padding)
        self.spatial_conv = nn.Conv2d(in_channels=out_channels, out_channels=spatial_channels, kernel_size=(self.Ks, 1), stride=1, padding=((self.Ks-1)//2, 0))

        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # X: batch, num_nodes, timestep, channel
        t0 = self.temporal1(X).permute(0, 3, 1, 2)  # t0 -> (batch, channel, nodes, timestep)

        # 1-D Spatial Conv
        t1 = self.spatial_conv(t0).permute(0, 2, 3, 1)  # (batch, channel, nodes, timestep) -> (batch, nodes, timestep, channel)
        t2 = F.relu(t1)
        t3 = self.temporal2(t2)  # t3: batch, nodes, timestep, channel
        return self.batch_norm(t3)


class STConv(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, Ks, Kt):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STConv, self).__init__()
        self.spatial_channel = 16
        self.temporal_channel = 64
        # self.Cheb = ChebPoly
        self.Kt = Kt  # Kernel size of temporal conv
        self.Ks = Ks
        self.block1 = STConvBlock(in_channels=num_features, out_channels=self.temporal_channel,
                                  spatial_channels=self.spatial_channel, num_nodes=num_nodes,
                                  in_timesteps=num_timesteps_input, Ks=Ks)
        self.block2 = STConvBlock(in_channels=self.temporal_channel, out_channels=self.temporal_channel,
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
        out1 = self.block1(X)
        out2 = self.block2(out1)
        out3 = self.last_temporal(out2)
        out4 = self.fully1(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # out5 = self.fully2(F.relu(out4))
        return out4
