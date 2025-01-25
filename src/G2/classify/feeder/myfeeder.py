# sys
import numpy as np
import pickle

# torch
import torch

# visualization
import time

# operation
from . import tools


class Feeder():  # 用于我们算法预测的时候提取数据,接收第一组传来的datas,此时datas代表一个片段

    def __init__(self,
                 datas,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        # output pre_datas shape (N, C, T, V, M)
        self.N = 1  # sample 样本的数量
        self.C = 3  # channel 通道数
        self.T = 300  # frame 帧数
        self.V = 17  # joint 关键点数量
        self.M = 1  # person
        self.num_person_in = 1
        self.datas = datas

    def process_data(self):  # 此处datas为一个有效片段的所有帧
        # pre_datas: N C V T M
        data_numpy = np.zeros((self.C, 300, self.V, self.num_person_in))
        for frame_index in range(0, len(self.datas)):  # 遍历每一帧
            for point in range(0, 17):
                data_numpy[0, frame_index, point, 0] = self.datas[frame_index][point][0]  # 从0开始，每隔一个元素去一个，即2个channel是x，y？
                data_numpy[1, frame_index, point, 0] = self.datas[frame_index][point][1]  # y
                data_numpy[2, frame_index, point, 0] = self.datas[frame_index][point][2]  # score
        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0
        # sort by score todo 是否有必要性
        # sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        # for t, s in enumerate(sort_index):
        #     data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
        #                                                                0))
        # data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        # match poses between 2 frames 两帧之间比对pose
        # if self.pose_matching:
        #     data_numpy = tools.openpose_match(data_numpy)
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        return data_numpy[np.newaxis, :]
