import math

import cv2

from src.utils import util
from src.G3.rules.dig import armDetect
from src.utils.logger import *


def calculate_average_hand_pose(hand_pose, start=0, end=22):
    total_x = 0
    total_y = 0
    num_valid_points = 0

    for point in hand_pose[start:end]:
        if point != -1:
            total_x += point[0]  # 假设关键点的坐标是一个元组，第一个元素是 x 坐标
            total_y += point[1]  # 假设关键点的坐标是一个元组，第二个元素是 y 坐标
            num_valid_points += 1

    if num_valid_points == 0:
        return None  # 如果没有有效的关键点，则返回 None

    average_x = total_x / num_valid_points
    average_y = total_y / num_valid_points

    return (average_x, average_y)  # 返回平均坐标的元组


def count_valid_hand_points(hand_pose, start=0, end=22):
    num_valid_points = sum(1 for point in hand_pose[start:end] if point != -1)
    return num_valid_points


def detect_wrist_angle(image, candidate, person, hand_pose):
    left_mid, left_wrist = None, None
    right_mid, right_wrist = None, None
    mistake = False
    try:
        left_mid = util.num2pos(6, candidate, person)
        left_wrist = util.num2pos(7, candidate, person)
    except:
        pass

    try:
        right_mid = util.num2pos(3, candidate, person)
        right_wrist = util.num2pos(4, candidate, person)
    except:
        pass

    left_hand_points = hand_pose[0:22]
    right_hand_points = hand_pose[22:44]

    left_valid = count_valid_hand_points(left_hand_points) > 11 and left_mid is not None
    right_valid = count_valid_hand_points(right_hand_points) > 11 and right_mid is not None

    # 定义颜色变量
    point_color = (255, 255, 255)
    left_arm_color = (244, 255, 84)
    if left_valid:
        left_hand = calculate_average_hand_pose(left_hand_points)
        left_angle = armDetect.detect_line([left_hand, left_wrist, left_mid])
        Log.info(f"左腕关键点有效 手腕角度{left_angle}")
        if left_angle < 130:
            mistake = True
            cv2.circle(image.get_self(), (int(left_hand[0]), int(left_hand[1])), 3, point_color, -1)  # 将手心圆点改为白色
            cv2.circle(image.get_self(), (int(left_mid[0]), int(left_mid[1])), 3, point_color, -1)  # 将手肘圆点改为蓝色
            cv2.circle(image.get_self(), (int(left_wrist[0]), int(left_wrist[1])), 3, point_color, -1)  # 将手腕圆点改为蓝色
            cv2.line(image.get_self(), (int(left_wrist[0]), int(left_wrist[1])), (int(left_mid[0]), int(left_mid[1])),
                     left_arm_color, 1)  # 将手腕到手肘的连线改为绿色
            cv2.line(image.get_self(), (int(left_wrist[0]), int(left_wrist[1])), (int(left_hand[0]), int(left_hand[1])),
                     left_arm_color, 1)  # 将手腕到手心的连线改为绿色
            util.draw_wrong_place(image.get_self(), left_wrist[0], left_wrist[1])

    if right_valid:
        right_hand = calculate_average_hand_pose(right_hand_points)
        right_angle = armDetect.detect_line([right_hand, right_wrist, right_mid])
        Log.info(f"右腕关键点有效 手腕角度{right_angle}")
        if right_angle < 130:
            mistake = True
            cv2.circle(image.get_self(), (int(right_hand[0]), int(right_hand[1])), 3, point_color, -1)
            cv2.circle(image.get_self(), (int(right_mid[0]), int(right_mid[1])), 3, point_color, -1)
            cv2.circle(image.get_self(), (int(right_wrist[0]), int(right_wrist[1])), 3, point_color, -1)
            cv2.line(image.get_self(), (int(right_wrist[0]), int(right_wrist[1])), (int(right_mid[0]), int(right_mid[1])),
                     left_arm_color, 1)
            cv2.line(image.get_self(), (int(right_wrist[0]), int(right_wrist[1])), (int(right_hand[0]), int(right_hand[1])),
                     left_arm_color, 1)
            util.draw_wrong_place(image.get_self(), right_wrist[0], right_wrist[1])

    return mistake
