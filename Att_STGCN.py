import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))
        self.reset_parameters()

    def reset_parameters(self):
        # only tensor with 2 or more dims can apply xavier_uniform_
        nn.init.normal_(self.W1)
        nn.init.xavier_uniform_(self.W2, gain=1)
        nn.init.normal_(self.W3)
        nn.init.xavier_uniform_(self.bs, gain=1)
        nn.init.xavier_uniform_(self.Vs, gain=1)

    def forward(self, x):
        '''
        :param x: (batch_size, Nodes(N), Channel, T)
        :return: (batch_size,N,N)
        '''

        lhs0 = torch.einsum('bnct,t->bnc', x, self.W1)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        lhs = torch.matmul(lhs0, self.W2)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class Temporal_Attention_layer(nn.Module):
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))
        self.reset_parameters()

    def reset_parameters(self):
        # only tensor with 2 or more dims can apply xavier_uniform_
        nn.init.normal_(self.U1)
        nn.init.xavier_uniform_(self.U2, gain=1)
        nn.init.normal_(self.U3)
        nn.init.xavier_uniform_(self.be, gain=1)
        nn.init.xavier_uniform_(self.Ve, gain=1)

    def forward(self, x):
        '''
        :param x: (batch_size, Nodes(N), Channel, T)
        :return: (batch_size, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


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

        # 这里修改了时间卷积以及Res连接的方式. here GLU gate works as activation function
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
        self.temporal_attention = Temporal_Attention_layer(in_channels, num_nodes, in_timesteps)
        self.spatial_attention = Spatial_Attention_layer(in_channels, num_nodes, in_timesteps)
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
        T_Att = self.temporal_attention(X.permute(0, 1, 3, 2))
        x_TAtt = torch.einsum('bnct,btu->bncu', X.permute(0, 1, 3, 2), T_Att)  # u:=t
        S_Att = self.spatial_attention(x_TAtt)  # S_Att: (batch, Nodes, Nodes)

        # bnct(x_TAtt)->bntc(temporal layer input)
        t0 = self.temporal1(x_TAtt.permute(0, 1, 3, 2)).permute(0, 3, 1, 2)  # t0: bntc -> bcnt

        if self.Ks - 1 < 0:
            raise ValueError(
                f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')

        # ChebGraphConv
        Cheb_SAtt = torch.einsum('kmn,bmn->kbmn', cheb, S_Att)  # k:Ks; b:batch; m==n==Nodes;
        Cheb_SAtt_x = torch.einsum('kbmn,bcnt->kbcmt', Cheb_SAtt, t0)  # c:channel; t:timestep;
        t2 = torch.einsum('kco,kbcmt->kbomt', self.Theta1, Cheb_SAtt_x)
        t3 = F.relu(torch.einsum('kbcnt->bcnt', t2).permute(0, 2, 3, 1))  # t3: batch, nodes, timestep, channel
        t4 = self.temporal2(t3)
        return self.batch_norm(t4)


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
