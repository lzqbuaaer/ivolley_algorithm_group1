# sys
import os
import numpy as np
import json
# torch
import torch
# operation
from . import tools


class Feeder_volley(torch.utils.data.Dataset):  # 用于 处理 训练数据
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' pre_datas, the shape of pre_datas should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 datas,  # 此时的datas为全为一个连续的有效片段
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=5,
                 num_person_out=2,
                 debug=False):
        self.debug = debug
        self.datas = datas
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):  # 加载原始数据
        # load file list
        self.sample_name = os.listdir(self.data_path)  # 返回data_path目录下的文件名，每一个视频对应一个文件吧

        # if self.debug:
        # self.sample_name = self.sample_name[0:2]

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)  # 将标签文件转化为字典

        sample_id = [name.split('.')[0] for name in self.sample_name]  # 获得每个sample的id
        self.label = np.array(
            [label_info[id]['label_index'] for id in sample_id])  # 根据sampleid来得到每一个sample的类别
        # has_skeleton = np.array(
        #   [label_info[id]['has_skeleton'] for id in sample_id])#根据sampleid得到每一个sample 的has_skeleton,判断是否有有效骨骼

        # ignore the samples which does not has skeleton sequence忽略掉没有骨骼序列的sample
        # if self.ignore_empty_sample:
        #     self.sample_name = [
        #         s for h, s in zip(has_skeleton, self.sample_name) if h
        #     ]
        #     self.label = self.label[has_skeleton]

        # output pre_datas shape (N, C, T, V, M)
        self.N = len(self.sample_name)  # sample 样本的数量
        self.C = 3  # channel 通道数
        self.T = 300  # frame 帧数
        self.V = 18  # joint 关键点数量
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):  # 得到具体每个sample的信息

        # output shape (C, T, V, M)
        # get pre_datas
        sample_name = self.sample_name[index]

        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        # for frame_info in video_info['pre_datas']:#每一个，frame_index，skeleton。。。。
        #     frame_index = frame_info['frame_index']
        #     for m, skeleton_info in enumerate(frame_info["skeleton"]):
        #         if m >= self.num_person_in:#可能有很多个人
        #             break

        for frame_index in range(0,self.datas.lenth):#遍历每一帧
            data_numpy[0, frame_index, :, 0] = self.datas[frame_index][0::3] #x # 从0开始，每隔一个元素去一个，即2个channel是x，y？
            data_numpy[1, frame_index, :, 0] = self.datas[frame_index][1::3] #y
            data_numpy[2, frame_index, :, 0] = self.datas[frame_index][2::3] #z

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        label = video_info['label_index']
        assert (self.label[index] == label)

        # pre_datas augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # sort by score todo 是否有必要性
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                       0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        # match poses between 2 frames 两帧之间比对pose
        if self.pose_matching:
            data_numpy = tools.openpose_match(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.label >= 0))
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.label >= 0))
        return tools.calculate_recall_precision(self.label, score)
