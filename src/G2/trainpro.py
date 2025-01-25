import json

import os
import src.G2.classify.tools.ivolley_gendata as gendata
from src.G2.classify.processor.recognition import REC_Processor as Processor
import torch
from src.G2.classify.feeder import feeder
from src.G2.classify.tools.ivolley_gendata import dataprocesser


def dataprepare():
    print("准备好了开始喽")
    dataPro = dataprocesser()
    folder = './json'
    filenames = os.listdir(folder)
    print(len(filenames))
    num = 0
    for file_name in filenames:
        num += 1
        print(num)
        print("files in json:" + folder)
        print("这次运行的文件名称是：", file_name)
        file_path = 'json/{}'.format(file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)
        actions = data['actions']
        keypoints = data['frames']
        for keypoint in keypoints:  # 为每个二维坐标添加置信度作为三维
            point = keypoint['points']
            score = keypoint['scores']
            point_3d = [[po[0], po[1], score[i]] for i, po in enumerate(point)]
            keypoint['points'] = point_3d
        for action in actions:  # 一个片段一个片段处理
            keypoints = keypoints[action['start']:action['end']]
            poses = []
            for i in keypoints:
                poses.append(i['points'])
            if len(poses) > 300:  # 每次传入一个sample
                dataPro.ivolley_gendata(poses[0:300], file_name)
            else:
                dataPro.ivolley_gendata(poses, file_name)

        print("{}文件执行完毕，继续".format(file_name))
    dataPro.genall()
    print("所有文件执行完毕，并已生成标签")


def trainangle():
    p = Processor('train', argv=['-c', 'classify/config/st_gcn/ivolley/train2.yaml'])
    p.start()


def judgeangle():
    p = Processor('predict', argv=['-c', 'classify/config/st_gcn/ivolley/test2.yaml'])
    folder = './test_video'
    filenames = os.listdir(folder)
    print("需要进行角度标签的文件数：", len(filenames))
    for file_name in filenames:
        labels = []  # 这一个视频所有片段的标签
        print("这次检测角度的json文件为：", file_name)
        file_path = 'test_video/{}'.format(file_name)
        with open(file_path, 'r') as file:
            datas = json.load(file)
        actions = datas['actions']
        keypoints = datas['frames']
        for keypoint in keypoints:
            point = keypoint['points']
            score = keypoint['scores']
            point_3d = [[po[0], po[1], score[i]] for i, po in enumerate(point)]
            keypoint['points'] = point_3d
            # print(keypoint)
        for action in actions:  # 一个片段一个片段处理
            keypoints = keypoints[action['start']:action['end']]
            poses = []
            for i in keypoints:
                poses.append(i['points'])
            if len(poses) > 300:  # 每次传入一个sample
                p.load_predict_data(poses[0:300])
            else:
                p.load_predict_data(poses)
            ans = p.predict()
            labels.append(ans[0])  # 返回的是一个片段的
            print(ans)
        print("labels:")
        print(labels)


def test_please_accurate():
    rights = 0
    errors = 0
    print("测试开始了！准点求求了")
    p = Processor('predict', argv=['-c', 'classify/config/st_gcn/ivolley/test2.yaml'])
    folder = './test_json'  # 更换为测试集的文件夹
    filenames = os.listdir(folder)
    for file_name in filenames:
        print("这次测试的json文件为：", file_name)
        file_path = 'test_json/{}'.format(file_name)
        parts = file_name.split('_')
        desired_part = parts[1].split('.')[0]
        with open(file_path, 'r') as file:
            datas = json.load(file)
        actions = datas['actions']
        keypoints = datas['frames']
        for keypoint in keypoints:
            point = keypoint['points']
            score = keypoint['scores']
            point_3d = [[po[0], po[1], score[i]] for i, po in enumerate(point)]
            keypoint['points'] = point_3d
            # print(keypoint)
        for action in actions:  # 一个片段一个片段处理
            keypoints = keypoints[action['start']:action['end']]
            poses = []
            for i in keypoints:
                poses.append(i['points'])
            if len(poses) > 300:  # 每次传入一个sample
                p.load_predict_data(poses[0:300])
            else:
                p.load_predict_data(poses)
            ans = p.predict()
            print("这个片段正确的标签为：" + str(desired_part) + "，本次模型预测的标签为：" + str(ans[0]))
            # 愚蠢计数
            if str(ans[0]) == str(desired_part):
                rights += 1
            else:
                errors += 1
    accrate_rate = (rights / (rights + errors)) * 100
    print("这个模型的准确率为" + str(accrate_rate) + "%")


def forwbx(datas):
    labels = []  # 这一个视频所有片段的标签
    p = Processor('predict', argv=['-c', 'classify/config/st_gcn/ivolley/test.yaml'])
    actions = datas['actions']
    keypoints = datas['frames']
    for keypoint in keypoints:
        point = keypoint['points']
        score = keypoint['scores']
        point_3d = [[po[0], po[1], score[i]] for i, po in enumerate(point)]
        keypoint['points'] = point_3d
        # print(keypoint)
    for action in actions:  # 一个片段一个片段处理
        keypoints = keypoints[action['start']:action['end']]
        poses = []
        for i in keypoints:
            poses.append(i['points'])
        if len(poses) > 300:  # 每次传入一个sample
            p.load_predict_data(poses[0:300])
        else:
            p.load_predict_data(poses)
        ans = p.predict()
        labels.append(ans[0])  # 返回的是一个片段的
        print(ans)
    print("labels:")
    print(labels)
    return labels


if __name__ == '__main__':
    test_please_accurate()
