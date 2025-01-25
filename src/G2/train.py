import os
from src.media.Video import Video

from src.G2.classify.processor.recognition import REC_Processor as Processor

from src.G2.classify.tools.ivolley_gendata import dataprocesser


def data_prepare():  # 用于处理videos数据
    dataPro = dataprocesser()
    src_path = r"./videos/"
    sample_name = os.listdir('videos')  # 返回目录下视频名
    for fileName in sample_name:
        print("fileName:" + fileName)
        # for fileName in ["1_4.mp4","2_4.mp4","3_4.mp4","4_4.mp4","5_4.mp4"]:
        videoPath = os.path.join(src_path, fileName)
        video = Video(videoPath)
        datas = video.preProcess()
        actions = datas['actions']
        keypoints = datas['frames']
        dataPro.num = 0  # 标记同一个视频内的不同片段
        for keypoint in keypoints:
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
                dataPro.ivolley_gendata(poses[0:300], fileName)
            else:
                dataPro.ivolley_gendata(poses, fileName)
        #  return {"actions": self.action, "keypoints": keypoints}
    # dataPro.genall()


def train():
    p = Processor('train', argv=['-c', 'classify/config/st_gcn/ivolley/train.yaml'])
    p.start()


def classifyForGroup2():  # 用于在已有all，label的情况下测试
    p = Processor('test', argv=['-c', 'classify/config/st_gcn/ivolley/test.yaml'])


def classify():  # 用于处理videos数据，与第一组联动
    p = Processor('predict', argv=['-c', 'classify/config/st_gcn/ivolley/test.yaml'])
    src_path = r"./videos/"
    sample_name = os.listdir('videos')  # 返回目录下视频名
    for fileName in sample_name:
        labels = []  # 这一个视频所有片段的标签
        videoPath = os.path.join(src_path, fileName)
        video = Video(videoPath)
        datas = video.preProcess()
        actions = datas['actions']
        keypoints = datas['frames']
        for keypoint in keypoints:
            point = keypoint['points']
            score = keypoint['scores']
            point_3d = [[po[0], po[1], score[i]] for i, po in enumerate(point)]
            keypoint['points'] = point_3d
            print(keypoint)
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



if __name__ == '__main__':
    train()
