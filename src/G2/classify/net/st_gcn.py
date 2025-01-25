import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.G2.classify.net.utils.tgcn import ConvTemporalGraphical
from src.G2.classify.net.utils.graph import Graph


class Model(nn.Module):  # 网络本身--结合教程细看源码
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input pre_datas
        num_class (int): Number of classes for the classification task 有几类
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,代表样本
            :math:`T_{in}` is a length of input sequence,帧数
            :math:`V_{in}` is the number of graph nodes,关键点数
            :math:`M_{in}` is the number of instance in a frame. 人数
    """

    # 网络输入是（N,C,T,V,M）
    # N 视频个数
    # C = 3 (X,Y,S)代表一个点的信息(位置+预测的可能性)
    # T = 300一个视频的帧数paper规定是300帧，不足的重头循环，多的clip
    # V 18 根据不同的skeleton获得的节点数而定，coco是18个节点
    # M = 2 人数，paper中将人数限定在最大2个人

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        # 这部分代码加载了一个图数据，并将其转换为张量 A。
        # 该张量被注册到模型的缓冲区中，以确保在模型的状态字典中进行持久化保存。
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # 以下是网络的结构
        # 一个输入层的batchNorm(接受的通道数是in_channels#3 * A.size(1)#18
        # 模型的输入是一个(N,C,T,V,M)的tensor第二部分由10层st_gcn层构成最后加一层全连接层
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # st_gcn_networks是一个由多个st_gcn组成的模块列表
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))
        # 这部分代码初始化了一个名为 st_gcn_networks 的模块列表，其中包含了多个 st_gcn 模块。
        # 每个 st_gcn 模块都有不同的输入和输出通道数，以及其他参数。
        # initialize parameters for edge importance weighting
        # 如果需要对边进行重要性权重设置，会初始化一个权重列表。todo 根据动作设置合适的权重
        # 否则，默认为每个 st_gcn 模块设置相同的权重。
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        ## 最后一层全连接层，输出和种类数有关
        # 最后，定义了一个卷积层用于模型的输出，将输入通道数为 256 的特征映射转换为最终的类别数。
        # fcn for prediction
        self.fcn = nn.Conv2d(256, 5, kernel_size=1)

    def forward(self, x):  # 整个Moule的forward函数

        # pre_datas normalization
        N, C, T, V, M = x.size()  # 网络的输入 获取输入张量 x 的维度信息
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        # 这行代码中的 permute 函数用于交换张量的维度，0, 4, 3, 1, 2 是维度的顺序。
        # 这里的目的是将张量 x 的维度重新排列，使得它变为 N，M，V，C，T 的顺序。
        # contiguous() 函数的作用是使得张量在内存中是连续存储的，这是一些操作的要求。
        x = x.view(N * M, V * C, T)
        # view 函数被用来改变张量的形状，这里将张量 x 变形为大小为 (N * M) × (V * C) × T 的三维张量。
        x = self.data_bn(x)  # 输入层的batchNorm（V*C）
        # 这行代码调用了模型中的 data_bn 函数，对 x 进行了数据的批量归一化操作
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        ## 注意，这里网络输入将 N, C, T, V, M整合成了N，C，T，V。
        # 将batch和person_num维度整合了一起

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # pre_datas normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence pre_datas
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output pre_datas in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
