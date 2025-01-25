import os
import sys
import pickle

import numpy as np

from numpy.lib.format import open_memmap

from src.G2.classify.feeder.feeder_volley import Feeder_volley

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

toolbar_width = 30


class dataprocesser():
    def __init__(self):
        self.alldatas = []  # 存所有的片段
        self.labels = []
        self.samplename = []
        self.num = 0

    # 注，每次传入本脚本的datas为一个完整的切分动作,name为当前处理视频的名字
    # 这个脚本用于将第一组传给我们的datas数据按照一个切分一个文件的方式保存到datas文件夹中
    def ivolley_gendata(self, datas, name):

        C = 3  # channel 通道数
        T = 300  # frame max帧数
        V = 17  # joint 关键点数量
        M = 1  # person

        datas = np.array(datas)
        data_numpy = np.zeros((C, T, V, M))

        for frame_index in range(0, len(datas)):  # 遍历每一帧
            for i in range(0, 17):
                data_numpy[0, frame_index, i, 0] = datas[frame_index][i][0]  # x # 从0开始，每隔一个元素去一个，即2个channel是x，y？
                data_numpy[1, frame_index, i, 0] = datas[frame_index][i][1]  # y
                data_numpy[2, frame_index, i, 0] = datas[frame_index][i][2]  # z

        # 使数据中心化方便训练
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))

        # match poses between 2 frames todo 两帧之间比对pose，不确定什么作用,此处被删了
        i = name.split('.')[0]
        data_out_path = '{}/{}_{}_{}.npy'.format("pre_datas", i.split('_')[1], name.split('.')[0], self.num)  # 这个文件的输出地址
        self.num += 1
        fp = open_memmap(
            data_out_path,
            dtype='float32',
            mode='w+',
            shape=(3, T, 17, 1))  # 创建新的输出文件，可以理解为数组形式，此处每次只有一个sample


        self.samplename.append(name)
        fp[:] = data_numpy[:]

    # # 用于把所有的sample打包到一个npy文件里，label打包到一个pck文件里
    def genall(self):
        # # load file list
        sample_name = os.listdir('pre_datas')  # 返回data_path目录下的文件名，每一个视频对应一个文件吧
        data_out_path = '{}/{}.npy'.format('datas', 'all')
        fp = open_memmap(
            data_out_path,
            dtype='float32',
            mode='w+',
            shape=(len(sample_name), 3, 300, 17, 1))  # 创建新的输出文件，可以理解为数组形式，此处每次只有一个sample
        mark = 0
        self.labels = []
        for each in sample_name:
            real_data = np.load('{}/{}'.format('pre_datas', each), allow_pickle=True)  # 类型是numpy array
            fp[mark, :, :, :, :] = real_data[np.newaxis, :]
            mark += 1
            self.labels.append(each.split('.')[0].split('_')[0])

        label_out_path = '{}/{}_label.pkl'.format('label', 'train')


        # 训练的时候可以生成新的文件
        with open(label_out_path, 'wb') as f:
            pickle.dump((self.samplename, list(self.labels)), f)  # 打包lable
