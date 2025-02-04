import math
import os.path
from pathlib import Path
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer

from settings import *
from moviepy.video.io.VideoFileClip import VideoFileClip
import settings
from src.media import tools_3d
from src.media.Image import Image
from src.utils.logger import Log

REDUN_FLAG = 0
TEMP_FLAG = -1
NEEDED_FLAG = 1
HEAT_BALL_DISTANCE = 0.75
wholebody_estimation = MMPoseInferencer(
    pose2d=os.path.join(settings.PythonPath, settings.pose2dPath),
    pose2d_weights=os.path.join(settings.AlgorithmPath, 'model',
                                'hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth')
)


class Video:
    output_num = "_mark"
    # output_format = output_num + ".mp4"
    output_format = output_num + ".avi"

    def __init__(self, url):
        self.dumped = False
        self.frames = []
        self.gotBall = []
        self.gotPerson = []
        self.flags = []
        self.action = []
        self.url = url
        self.videoCapture = cv2.VideoCapture(url)  # 要解析的视频对象
        self.fps = int(self.videoCapture.get(cv2.CAP_PROP_FPS))  # 原视频帧率
        self.total = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.gap = 1
        self.neighbor = self.fps // 10
        self.width = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.output_path = os.path.join(settings.AlgorithmPath, settings.VideoOutputPath,
                                        Path(url).stem) + Video.output_format
        Log.info(f"开始解析视频 目标路径 : {self.output_path}")

        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps,
                                      (self.width, self.height))
        if not self.writer.isOpened():
            print("视频写入器无法打开，请检查输出路径或编码格式")
        self.blank_frame = 255 * np.ones((self.height, self.width, 3), np.uint8)  # 创建一个白色的帧，如果需要黑色帧可以使用0 * np.ones
        while True:
            ret = self.videoCapture.grab()
            if ret:
                ret, frame = self.videoCapture.retrieve()
                image = Image(frame)
                self.frames.append(image)
            else:
                break

    def markBall(self, pre_roll=10, post_roll=10):
        cnt = 0
        Log.info("-------------------- mark ball -----------------------")
        for i, frame in enumerate(self.frames):
            if i % self.gap == 0:
                ball = frame.mark_ball()
                print(f"In frame{i}, ball position:{ball}")
                self.gotBall.append(TEMP_FLAG if ball is not None else REDUN_FLAG)
                cnt = cnt + 1 if ball is not None else cnt


        detectLength = len(self.gotBall)

        if detectLength == 0 or cnt <= detectLength * 0.15:
            self.dumped = True
            return

        tmp = list()  # 前缀和
        tmp.append(int(self.gotBall[0] == TEMP_FLAG))
        for i in range(1, detectLength):
            tmp.append(int(self.gotBall[i] == TEMP_FLAG) + tmp[-1])

        halfLen = min(10, detectLength) // 2
        for i in range(halfLen, detectLength - halfLen):
            if tmp[i + halfLen] - tmp[i - halfLen] == 1 and self.gotBall[i] == TEMP_FLAG:
                self.gotBall[i] = REDUN_FLAG

        for i, flag in enumerate(self.gotBall):
            if flag == TEMP_FLAG:
                if i != 0 and i != detectLength - 1 and self.gotBall[i - 1] == TEMP_FLAG and self.gotBall[
                    i + 1] == TEMP_FLAG:
                    continue
                for j in range(max(i - pre_roll, 0), min(i + post_roll, detectLength - 1) + 1):  # range 左闭右开
                    if self.gotBall[j] == REDUN_FLAG:
                        self.gotBall[j] = NEEDED_FLAG
        for i, flag in enumerate(self.gotBall):
            if flag == TEMP_FLAG:
                self.gotBall[i] = NEEDED_FLAG
        for flag in self.gotBall:
            for i in range(self.gap):
                self.flags.append(flag)
        while len(self.flags) < len(self.frames):
            self.flags.append(self.flags[-1])  # 补全末尾

        cnt = 0
        for i in range(len(self.flags)):
            cnt += int(self.flags[i] != REDUN_FLAG)

        if cnt == 0:
            self.dumped = True

    def ballRebuild(self, kToContain=4, weightMethod='sq_dist'):
        Log.info("----------------------- ball rebuild --------------------------")

        def get_weight(cur, windowEnd, sublen, method='sq_dist'):
            if method == 'dist':
                return sublen // 2 - abs(windowEnd - sublen // 2 - cur) + 1
            elif method == 'sq_dist':
                return (sublen // 2 - abs(windowEnd - sublen // 2 - cur) + 1) ** 2
            elif method == 'exp_dist':
                return np.exp2(sublen // 2 - abs(windowEnd - sublen // 2 - cur) + 1)
            else:
                return 1

        first = 0
        found = False
        while first < len(self.flags):
            if self.flags[first] == NEEDED_FLAG:
                found = True
                break
            first += 1

        if not found:  # 没有要保留的帧，完全没检测到球
            return

        clips = list()  # 构建连续需要帧的左闭右开区间列表
        last = first + 1
        while last < len(self.flags):
            if self.flags[last] == REDUN_FLAG and self.flags[last - 1] == NEEDED_FLAG:
                clips.append([first, last])
            elif self.flags[last] == NEEDED_FLAG and self.flags[last - 1] == REDUN_FLAG:
                first = last
            last += 1
        if self.flags[-1] == NEEDED_FLAG:
            clips.append([first, last])

        for clip in clips:
            begin = clip[0]
            end = clip[1]

            balls = []
            for frame in self.frames[begin:end]:
                balls.append(frame.ball_center)

            def detect_outliers_zscore_2d(_data, threshold=1.8):
                non_outliers_mask = np.all(_data != [-1, -1], axis=1)
                data_without_negatives = _data[non_outliers_mask]
                z_scores_dim1 = (data_without_negatives[:, 0] - np.mean(data_without_negatives[:, 0])) / np.std(
                    data_without_negatives[:, 0])
                z_scores_dim2 = (data_without_negatives[:, 1] - np.mean(data_without_negatives[:, 1])) / np.std(
                    data_without_negatives[:, 1])
                outliers_dim1 = np.where(np.abs(z_scores_dim1) > threshold)[0]
                outliers_dim2 = np.where(np.abs(z_scores_dim2) > threshold)[0]
                # 合并异常点索引
                outliers = non_outliers_mask.nonzero()[0][np.union1d(outliers_dim1, outliers_dim2)]
                return outliers

            data = np.array(balls)
            outliers = detect_outliers_zscore_2d(data)

            # 将异常点替换为[-1, -1]
            for idx in outliers:
                balls[idx] = [-1, -1]
            # plt.scatter(data[:, 0], data[:, 1], c='blue', label='Normal')
            # plt.scatter(data[outliers, 0], data[outliers, 1], c='red', label='Anomalies')
            # plt.legend()
            # plt.show()

            # non_outliers = np.setdiff1d(np.arange(len(data)), outliers)
            # balls = data[non_outliers].tolist()

            first = 0
            last = 0
            sub_len = -1
            valid_cnt = 0
            while last < len(balls):
                if valid_cnt < kToContain:
                    valid_cnt += balls[last][1] != -1
                    last += 1
                if valid_cnt == kToContain:
                    sub_len = max(sub_len, last - first + 1)
                    valid_cnt -= balls[first][1] != -1
                    first += 1
                    # if valid_cnt == kToContain - 1:
                    #     sub_len = max(sub_len, last - first + 1)

            if sub_len != -1 and valid_cnt + 1 == kToContain:
                sub_len = max(sub_len, last - first + 1)  # last == len, first - 1处有球

            if sub_len == -1:  # 数据太少，线性插值
                i_detected = list()
                x_detected = list()
                y_detected = list()
                for i, loc in enumerate(balls):
                    if loc != [-1, -1]:
                        i_detected.append(i)
                        x_detected.append(loc[0])
                        y_detected.append(loc[1])

                x_coefficients = np.polyfit(np.asarray(i_detected), np.asarray(x_detected), 1)
                y_coefficients = np.polyfit(np.asarray(i_detected), np.asarray(y_detected), 1)
                x_poly1 = np.poly1d(x_coefficients)
                y_poly1 = np.poly1d(y_coefficients)

                for i, frame in enumerate(self.frames[begin:end]):
                    if i not in i_detected:
                        self.frames[begin + i].ball_center = [max(0, min(self.width, x_poly1(i))),
                                                              max(0, min(self.height, y_poly1(i)))]
                return

            pred_dict_x = dict()
            pred_dict_y = dict()

            i_detected = list()
            x_detected = list()
            y_detected = list()
            i_not_detected = list()
            x_not_detected = list()
            y_not_detected = list()
            # Stage 1
            for i in range(sub_len):
                if tuple(balls[i]) != (-1, -1):
                    i_detected.append(i)
                    x_detected.append(balls[i][0])
                    y_detected.append(balls[i][1])
                else:
                    i_not_detected.append(i)

            x_coefficients = np.polyfit(np.asarray(i_detected), np.asarray(x_detected), 2)  # 拟合二次多项式
            y_coefficients = np.polyfit(np.asarray(i_detected), np.asarray(y_detected), 2)
            x_poly = np.poly1d(x_coefficients)  # 创建多项式对象
            y_poly = np.poly1d(y_coefficients)

            for num in i_not_detected:
                weight = get_weight(num, sub_len - 1, sub_len, weightMethod)
                if num in pred_dict_x.keys():
                    pred_dict_x[num].append((weight, x_poly(num)))
                    pred_dict_y[num].append((weight, y_poly(num)))
                else:
                    pred_dict_x[num] = [(weight, x_poly(num))]
                    pred_dict_y[num] = [(weight, y_poly(num))]

            # Stage 2
            detected_first = 0
            not_detected_first = 0
            for i in range(sub_len, len(balls)):
                if tuple(balls[i - sub_len]) != (-1, -1):
                    detected_first += 1
                else:
                    not_detected_first += 1

                if tuple(balls[i]) != (-1, -1):
                    i_detected.append(i)
                    x_detected.append(balls[i][0])
                    y_detected.append(balls[i][1])
                else:
                    i_not_detected.append(i)

                x_coefficients = np.polyfit(np.asarray(i_detected[detected_first:]),
                                            np.asarray(x_detected[detected_first:]), 2)
                y_coefficients = np.polyfit(np.asarray(i_detected[detected_first:]),
                                            np.asarray(y_detected[detected_first:]), 2)
                x_poly = np.poly1d(x_coefficients)  # 创建多项式对象
                y_poly = np.poly1d(y_coefficients)

                for j in range(not_detected_first, len(i_not_detected)):
                    num = i_not_detected[j]
                    weight = get_weight(num, i, sub_len, weightMethod)
                    if num in pred_dict_x.keys():
                        pred_dict_x[num].append((weight, x_poly(num)))
                        pred_dict_y[num].append((weight, y_poly(num)))
                    else:
                        pred_dict_x[num] = [(weight, x_poly(num))]
                        pred_dict_y[num] = [(weight, y_poly(num))]

            # Stage 3
            for index, num in enumerate(i_not_detected):
                weights, val = zip(*pred_dict_x[num])
                res = np.average(val, weights=weights)
                x_not_detected.append(max(0, min(self.width, res)))
                weights, val = zip(*pred_dict_y[num])
                res = np.average(val, weights=weights)
                y_not_detected.append(max(0, min(self.height, res)))
                self.frames[begin + num].ball_center = [x_not_detected[index], y_not_detected[index]]
        valid = 0
        redis = 0
        for frame in self.frames:
            ball_xy = frame.get_ball()
            if ball_xy[0] != -1:
                valid += 1
                redis += (abs(ball_xy[0] - ball_xy[2]) + abs(ball_xy[1] - ball_xy[3])) / 4
        redis /= valid
        Log.error(f"redis : {redis}")
        for frame in self.frames:
            if frame.get_ball()[0] == -1:
                frame.predict_by_redis(redis)

            ############################################################################################################
            # 用于调参
            # # 使用subplot分别绘制x和y坐标的散点图
            # plt.figure(figsize=(12, 6))
            # # 绘制x坐标的散点图
            # plt.subplot(2, 1, 1)
            # plt.title(f"kToContain={kToContain}, weightMethod={weightMethod}")
            # plt.scatter(i_detected, x_detected, marker='o', color='b')
            # plt.scatter(i_not_detected, x_not_detected, marker='o', color='r')
            # plt.grid(True)
            #
            # # 绘制y坐标的散点图
            # plt.subplot(2, 1, 2)
            # plt.scatter(i_detected, y_detected, marker='o', color='b')
            # plt.scatter(i_not_detected, y_not_detected, marker='o', color='r')
            # plt.grid(True)
            #
            # plt.tight_layout()  # 调整子图的布局，防止重叠
            # plt.show()
            #
            # for index, val in enumerate(i_not_detected):
            #     x = max(0, min(self.width, x_not_detected[index]))
            #     y = max(0, min(self.height, y_not_detected[index]))
            #
            #     balls[val] = (x, y)
            #
            # for i, frame in enumerate(self.frames[begin:end]):
            #     img = frame.get_self()
            #     cv2.circle(img, (int(balls[i][0]), int(balls[i][1])), 10,
            #                (0, 0, 255) if frame.ball_left == -1 else (255, 0, 0), -1)
            #     cv2.imshow("ball", img)
            #     cv2.waitKey(10)
            ############################################################################################################

        # for frame in self.frames:
        #     img = frame.get_self()
        #     x = int(frame.ball_center[0])
        #     y = int(frame.ball_center[1])
        #     if x != -1:
        #         cv2.circle(img, (x, y), 10,
        #                    (0, 0, 255) if frame.ball_left == -1 else (255, 0, 0), -1)
        #         cv2.imshow("ball", img)
        #         cv2.waitKey(33)

    def markWholeBody(self):
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

        def chose_master(predictions, ball_center):
            if len(predictions[0]) == 0:
                Log.error("0个人")
                return predictions[0][0], False
            elif len(predictions[0]) == 1:
                if predictions[0][0]['bbox_score'] < settings.MIN_BBOX_SCORE or predictions[0][0]['bbox_score'] > 0.999:
                    Log.error(f"1个人 bbox_score过低 : {predictions[0][0]['bbox_score']}")
                    return predictions[0][0], False
                Log.error(f"1个人 bbox_score达标 : {predictions[0][0]['bbox_score']}")
                return predictions[0][0], False
            dis_min = float('inf')
            ans = -1
            for index, prediction in enumerate(predictions[0]):
                if prediction['bbox_score'] < settings.MIN_BBOX_SCORE:
                    continue
                dis_new = calculate_distance(average_body_pos(prediction), ball_center)
                Log.warning(f"dis_min = {dis_min}")
                Log.warning(f"dis_new = {dis_new}")
                if dis_min > dis_new:
                    dis_min, ans = dis_new, index
            Log.error(f"choose = {dis_min} -- {ans}")
            if ans == -1:
                Log.error(f"多个人 bbox_score未达标 1 :{predictions[0][0]['bbox_score']}")
                return predictions[0][0], False
            Log.error(f"多个人 bbox_score达标 {predictions[0][ans]['bbox_score']}")
            return predictions[0][ans], ans > 0

        Log.info("-------------------- mark whole body ---------------------")
        result_generator = wholebody_estimation(self.url)
        results = [result for result in result_generator]
        [lefts, rights, angles, keyPoints3d] = tools_3d.process3d(self.url)
        keyPoints = []
        for index, result in enumerate(results):
            self.flags[index] = NEEDED_FLAG
            body_points, scores, hands, hands_scores = [], [], [], []
            chosen_result, multi = chose_master(result['predictions'], self.frames[index].ball_center)
            if multi:
                if index % 2 == 0:
                    angles[index] = 90
                else:
                    angles[index] = -90
            ret_scores = chosen_result['keypoint_scores']
            ret_keypoints = chosen_result['keypoints']
            forbid_left, forbid_right = False, False
            point_left, point_right = [5, 7, 9], [6, 8, 10]
            for i in range(5, 11):
                if i in point_left and ret_scores[i] < settings.MIN_POINT_SCORE:
                    forbid_left = True
                if i in point_right and ret_scores[i] < settings.MIN_POINT_SCORE:
                    forbid_right = True
            for i in range(17):
                if (forbid_left and i in point_left) or (forbid_right and i in point_right):
                    scores.append(ret_scores[i])
                    body_points.append([-1, -1])
                    continue
                x, y = ret_keypoints[i]
                scores.append(ret_scores[i])
                body_points.append([x, y])
            for i in range(91, 133):  # 91 - 112 | 113 - 132
                x, y = ret_keypoints[i]
                hands_scores.append(ret_scores[i])
                hands.append([x, y])
            keyPoints.append(
                {"flag": NEEDED_FLAG, "points": body_points, "scores": scores, "ball": self.frames[index].get_ball(),
                 "hands": hands, "hands_scores": hands_scores, "3d_points": keyPoints3d[index],
                 "3d_angle": angles[index]})
            self.frames[index].setkeyPoints(body_points)
        # for flag, frame in zip(self.flags, self.frames):
        #     if flag == NEEDED_FLAG:
        #         points, scores, hands, hand_scores = frame.mark_wholebody()
        #         # Log.info(f"hands scores {hand_scores}")
        #         # Log.info(f"body scores {scores}")
        #         if len(points) != 0 and len(scores) != 0 and len(hand_scores) != 0 and len(hands) != 0:
        #             keyPoints.append({"flag": NEEDED_FLAG, "points": points, "scores": scores, "ball": frame.get_ball(),
        #                               "hands": hands, "hands_scores": hand_scores})
        #             self.gotPerson.append(NEEDED_FLAG)
        #         else:
        #             keyPoints.append(
        #                 {"flag": REDUN_FLAG, "points": [], "scores": [], "ball": [], "hands": [], "hands_scores": []})
        #             self.gotPerson.append(REDUN_FLAG)
        #     else:
        #         keyPoints.append(
        #             {"flag": REDUN_FLAG, "points": [], "scores": [], "ball": [], "hands": [], "hands_scores": []})
        #         self.gotPerson.append(REDUN_FLAG)
        # for index in range(len(self.frames)):
        #     if self.flags[index] == NEEDED_FLAG and self.gotPerson[index] == NEEDED_FLAG:
        #         self.flags[index] = NEEDED_FLAG
        #     else:
        #         self.flags[index] = REDUN_FLAG
        return keyPoints

    def average_neighboring_elements(self, arr):
        result = []
        current_group = [arr[0]]
        for i in range(1, len(arr)):
            if arr[i] - arr[i - 1] <= self.neighbor:
                current_group.append(arr[i])
            else:
                avg_value = sum(current_group) // len(current_group)
                result.append(avg_value)
                current_group = [arr[i]]
        if current_group:
            avg_value = sum(current_group) // len(current_group)
            result.append(avg_value)
        return result

    def cut_action(self):
        Log.info("-------------------- cut action ---------------------")
        parts = []
        is_continuous = False
        start = None
        Log.info(self.flags)
        for i, flag in enumerate(self.flags):
            if flag == NEEDED_FLAG:
                if not is_continuous:
                    start = i
                    is_continuous = True
            else:
                if is_continuous:
                    parts.append([start, i])
                    is_continuous = False
        if is_continuous:
            parts.append([start, len(self.flags)])

        for part in parts:
            hits = []
            borders = []
            for i in range(part[0], part[1]):
                dis = self.frames[i].get_distance()
                Log.error(f"dis = {dis}")
                if dis < HEAT_BALL_DISTANCE and dis != 0:
                    hits.append(i)
            if len(hits) == 0:
                self.action.append({"start": part[0], "hit": int((part[0] + part[1]) / 2), "end": part[1]})
                Log.info(f"hits :{hits}")
                Log.info(f"border : {borders}")
                return
            hits = self.average_neighboring_elements(hits)
            borders.append(part[0])
            for i in range(0, len(hits) - 1):
                borders.append((hits[i] + hits[i + 1]) // 2)
            borders.append(part[1] - 1)
            for i in range(0, len(borders) - 1):
                if i == len(borders) - 2:
                    end = borders[i + 1]
                else:
                    end = borders[i + 1] - 1
                self.action.append({"start": borders[i], "hit": hits[i], "end": end})
            Log.info(f"hits :{hits}")
            Log.info(f"border : {borders}")
        Log.info(f"action: {self.action}")

    def preProcess(self):
        self.markBall()
        if self.dumped:
            return None
        self.ballRebuild()
        for i, frame in enumerate(self.frames):
            print(f"In frame{i}, ball position:{frame.get_ball()}")
        keypoints = self.markWholeBody()
        self.cut_action()
        Log.info("-------------------- finish preprocess ---------------------")
        return {"actions": self.action, "frames": keypoints}

    def transform_numbers(self, nums, slow_range, slow_para):
        def slow_down(lst, index, length):
            result = lst.copy()
            for i in range(length):
                if abs(index - i) == 4 and result[i] < slow_range * 1 / 5:
                    result[i] = slow_range * 1 / 5
                if abs(index - i) == 3 and result[i] < slow_range * 2 / 5:
                    result[i] = slow_range * 2 / 5
                if abs(index - i) == 2 and result[i] < slow_range * 3 / 5:
                    result[i] = slow_range * 3 / 5
                if abs(index - i) == 1 and result[i] < slow_range * 4 / 5:
                    result[i] = slow_range * 4 / 5
            return result

        def find_non_consecutive_segments(arr, x):
            segments = []
            start = None
            for i, num in enumerate(arr):
                if num != 1:
                    if start is None:
                        start = i
                elif start is not None:
                    segments.append((start, i - 1, x))
                    start = None
            if start is not None:
                segments.append((start, len(arr) - 1, x))
            return segments

        output_nums = nums.copy()
        length = len(nums)
        for i in range(length):
            if nums[i] == slow_range:
                output_nums = slow_down(output_nums, i, length)
        for i in range(length):
            output_nums[i] = int(output_nums[i])
        return find_non_consecutive_segments(output_nums, slow_para)

    def slow_motion_flow(self, video_path, slow_intervals, output_path):
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: 无法打开视频文件")
            return

        # 获取视频的帧率和总帧数
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入对象
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (int(cap.get(3)), int(cap.get(4))))

        # 初始化变量
        current_frame = 0
        current_interval_index = 0
        start_frame, end_frame, slow_factor = slow_intervals[current_interval_index]

        # 读取第一帧
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, frame = cap.read()

            if ret:
                # 计算光流
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # 写入慢放的帧
                if current_frame >= start_frame and current_frame < end_frame:
                    for _ in range(slow_factor):
                        out.write(frame)
                else:
                    out.write(frame)

                # 检查是否到达当前时间段的结束帧
                if current_frame == end_frame - 1:
                    current_interval_index += 1
                    if current_interval_index < len(slow_intervals):
                        start_frame, end_frame, slow_factor = slow_intervals[current_interval_index]

                # 更新前一帧和当前帧
                prev_gray = frame_gray.copy()
                current_frame += 1
            else:
                break

        cap.release()
        out.release()
        Log.info(f"视频已生成 : {output_path}")

    def view_cut(self):
        for act in self.action:
            for index in range(act['start'], act['end'] + 1):
                self.writer.write(self.frames[index].get_self())
            for _ in range(10):
                self.writer.write(self.blank_frame)
        self.writer.release()

    def close(self):
        frame_nums = []
        for frame in self.frames:
            frame_nums.append(frame.get_mark())
        slow_parameter = self.transform_numbers(frame_nums, WRONG_MARK, 3)
        for frame in self.frames:
            self.writer.write(frame.get_self())
        self.writer.release()
        if len(slow_parameter) != 0:
            self.slow_motion_flow(self.output_path, slow_parameter, self.output_path.replace('.avi', '_slow.avi'))
        #     video = VideoFileClip(self.output_path.replace('.avi', '_slow.avi'))
        #     output_mp4 = self.output_path.replace('.avi', '_slow.mp4')
        #     video.write_videofile(output_mp4, codec='libx264', audio_codec='aac')
        #     os.remove(self.output_path.replace('.avi', '_slow.avi'))
        # else:  # 不需要慢放
        #     video = VideoFileClip(self.output_path)
        #     video.write_videofile(self.output_path.replace('.avi', '_slow.mp4'), codec='libx264', audio_codec='aac')
        # Log.info(f"视频已重构：{self.output_path.replace('.avi', '_slow.mp4')}")
        # os.remove(self.output_path)
        # return self.output_path.replace('.avi', '_slow.mp4')

        if len(slow_parameter) != 0:
            return self.output_path.replace('.avi', '_slow.avi')
        else:
            return self.output_path

