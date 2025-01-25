import os

import cv2

from src.G2 import trainpro
from src.media.Video import Video
from src.media.Image import Image
from src.G3.judge import judge_video, judge_image

def analyze_video(src_path, tag):
    video = Video(src_path)
    datas = video.preProcess()

    if datas is None:
        return set("未检测到球"), None

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_dir, "G2"))

    if tag == -1:
        labels = trainpro.forwbx(datas)
    else:
        actions = datas['actions']
        labels = [tag] * len(actions)
    os.chdir(os.path.join(current_dir, "G3"))
    msg, output_path = judge_video(labels, datas, video)
    os.chdir(current_dir)

    return msg, output_path

def analyze_image(src_path, tag):
    img = cv2.imread(src_path)
    image = Image(img, os.path.basename(src_path))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_dir, "G2"))

    if tag == -1:
        msg = set("图片暂不支持分类,请选择指定类别评判")
        output_path = None
    else:
        os.chdir(os.path.join(current_dir, "G3"))
        msg, output_path = judge_image(tag, None, image, src_path)

    return msg, output_path

