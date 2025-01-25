import math
import sys
from pathlib import Path

import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
from scipy.optimize import curve_fit

import settings
from settings import *

from src.models.ball import run
from src.models.common import DetectMultiBackend
from src.utils import util
from src.utils.logger import Log
from src.utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
device = select_device()
volleyball_model = DetectMultiBackend(str(ROOT / 'model/yolov5x.pt'), device=device, dnn=False,
                                      data='data/coco128.yaml')
wholebody_estimation = MMPoseInferencer('wholebody')


class Image:

    def __init__(self, frame, name=None):
        self.mark = CORRECT_MARK
        self.image = frame
        self.distance = 0
        self.keyPoints, self.candidate, self.subset, self.scores, self.hands, self.hands_scores = [], [], [], [], [], []
        self.ball_left = self.ball_right = self.ball_up = self.ball_down = -1
        self.ball_center = []  # [x, y]
        if name:
            self.output_path = os.path.join(settings.AlgorithmPath, settings.ImageOutputPath,
                                            Path(name).stem + "_mark.jpg")
            Log.info(f"开始解析图片 目标路径: {self.output_path}")

    def setkeyPoints(self, keypoints):
        self.keyPoints = keypoints

    def mark_ball(self):
        ball = run(volleyball_model, self.image)
        if ball:
            self.ball_left, self.ball_up, self.ball_right, self.ball_down = ball[:4]
            self.ball_center = [(self.ball_left + self.ball_right) / 2, (self.ball_down + self.ball_up) / 2]
        else:
            self.ball_left, self.ball_up, self.ball_right, self.ball_down = -1, -1, -1, -1
            self.ball_center = [-1, -1]
        return ball

    def mark_wholebody(self):
        def calculate_distance(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return distance

        def average_body_pos(body):
            points = body['keypoints']
            x_list, y_list = [], []
            for i in range(17):
                x_list.append(points[i][0])
                y_list.append(points[i][1])
            return np.mean(x_list), np.mean(y_list)

        def chose_master(predictions):
            if len(predictions[0]) == 0:
                Log.error("0个人")
                return None
            elif len(predictions[0]) == 1:
                if predictions[0][0]['bbox_score'] < settings.MIN_BBOX_SCORE or predictions[0][0]['bbox_score'] > 0.999:
                    Log.error(f"1个人 bbox_score过低 : {predictions[0][0]['bbox_score']}")
                    return None
                Log.error(f"1个人 bbox_score达标 : {predictions[0][0]['bbox_score']}")
                return predictions[0][0]
            dis_min = float('inf')
            ans = -1
            for index, prediction in enumerate(predictions[0]):
                if prediction['bbox_score'] < settings.MIN_BBOX_SCORE:
                    continue
                dis_new = calculate_distance(average_body_pos(prediction), self.ball_center)
                Log.warning(f"dis_min = {dis_min}")
                Log.warning(f"dis_new = {dis_new}")
                if dis_min > dis_new:
                    dis_min, ans = dis_new, index
            Log.error(f"choose = {dis_min} -- {ans}")
            if ans == -1:
                Log.error(f"多个人 bbox_score未达标 1 :{predictions[0][0]['bbox_score']}")
                return None
            Log.error(f"多个人 bbox_score达标 {predictions[0][ans]['bbox_score']}")
            return predictions[0][ans]

        ret = wholebody_estimation(self.image)
        result = chose_master(predictions=next(ret)['predictions'])
        if result is None:
            return [], [], [], []
        scores = result['keypoint_scores']
        keypoints = result['keypoints']
        # Log.info(f"bbox-score : {result['bbox_score']}")
        forbid_left, forbid_right = False, False
        point_left, point_right = [5, 7, 9], [6, 8, 10]
        for i in range(5, 11):
            if i in point_left and scores[i] < settings.MIN_POINT_SCORE:
                forbid_left = True
            if i in point_right and scores[i] < settings.MIN_POINT_SCORE:
                forbid_right = True
        for i in range(17):
            if (forbid_left and i in point_left) or (forbid_right and i in point_right):
                self.scores.append(scores[i])
                self.keyPoints.append([-1, -1])
                continue
            x, y = keypoints[i]
            self.scores.append(scores[i])
            self.keyPoints.append([x, y])
        for i in range(91, 133):  # 91 - 112 | 113 - 132
            x, y = keypoints[i]
            self.hands_scores.append(scores[i])
            self.hands.append([x, y])

        return self.keyPoints, self.scores, self.hands, self.hands_scores

    def get_distance(self):
        def point_to_segment_distance(point, segment):
            A, B = segment[0], segment[1]
            C = np.array(point)
            AB = np.array(B) - np.array(A)
            AC = C - np.array(A)

            # 计算参数t
            t = np.dot(AC, AB) / np.dot(AB, AB)

            # 如果t小于0，则取投影点为A；如果t大于1，则取投影点为B
            t = max(0, min(1, t))

            # 计算投影点D
            D = np.array(A) + t * AB

            # 计算点到投影点的距离
            distance = np.linalg.norm(C - D)

            return distance

        if self.ball_center[0] == -1:
            return 0
        arm_left = np.array([self.keyPoints[5], self.keyPoints[7], self.keyPoints[9]])
        arm_right = np.array([self.keyPoints[6], self.keyPoints[8], self.keyPoints[10]])

        distance_left = point_to_segment_distance((self.ball_center[0], self.ball_down),
                                                  [(self.keyPoints[7][0], self.keyPoints[7][1]),
                                                   (self.keyPoints[9][0], self.keyPoints[9][1])])
        distance_right = point_to_segment_distance((self.ball_center[0], self.ball_down),
                                                   [(self.keyPoints[8][0], self.keyPoints[8][1]),
                                                    (self.keyPoints[10][0], self.keyPoints[10][1])])
        arm_length_left = np.linalg.norm(arm_left[2] - arm_left[1]) + np.linalg.norm(arm_left[1] - arm_left[0])
        arm_length_right = np.linalg.norm(arm_right[2] - arm_right[1]) + np.linalg.norm(arm_right[1] - arm_right[0])

        self.distance = min(distance_left / arm_length_left, distance_right / arm_length_right)
        return self.distance

    def show_person(self):
        return util.draw_bodypose(self.image, self.candidate, self.subset)

    def get_ball(self):
        return [self.ball_left, self.ball_down, self.ball_right, self.ball_up]

    def predict_by_redis(self, redis):
        self.ball_left = self.ball_center[0] - redis
        self.ball_down = self.ball_center[1] + redis
        self.ball_right = self.ball_center[0] + redis
        self.ball_up = self.ball_center[1] - redis

    def get_self(self):
        return self.image

    def update_self(self, image):
        self.image = image

    def mark_wrong(self):
        self.mark = WRONG_MARK

    def get_mark(self):
        return self.mark

    def close(self):
        cv2.imwrite(self.output_path, self.image)
        Log.info("图片已生成")
        return self.output_path
