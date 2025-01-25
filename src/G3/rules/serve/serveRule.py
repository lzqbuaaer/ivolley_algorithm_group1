import math

import numpy as np

from src.G3.rules.dig import armDetect, armTorsoAngle, legDetect
from src.G3.rules.dig.ball_position import ball_position
from src.utils.logger import Log
from src.utils.util import arm_dis_ball, draw_messages, draw_wrong_place, point_dis_line

from src.utils import util
from copy import deepcopy
import cv2


def sum_rules(actionPoints, images, candidates, persons, balls, hit):
    mes, total_msg = set(), set()
    marks = []
    cnt = 0
    if len(images) == 1:  # 是图片
        if candidates[0] is None or persons[0] is None:
            mes.add("人体未被识别")
            return list(mes)
        else:
            try:
                if balls[0] is None:
                    mes.add("球未被识别")
            except Exception as e:
                mes.add("可能存在关键点识别不全的问题")
            return list(mes)
    else:  # 是视频
        keypoints_2d = [item["points"] for item in actionPoints]
        keypoints_3d = [item["3d_points"] for item in actionPoints]
        angle_3d = [item["3d_angle"] for item in actionPoints]
        detected = False
        for i in range(len(images)):
            marks.append(0)
            if candidates[i] is None or persons[i] is None:
                continue
            try:
                if balls[i] is None:
                    cnt += 1
                    Log.error("球未被识别")
                    if armDetect.digWithoutBall(images[i], candidates[i], persons[i]):
                        if not armDetect.detect_arm_bend(images[i], candidates[i], persons[i]):
                            marks[i] = 1
                            mes.add("手臂没有伸直")
            except Exception as e:
                Log.debug("可能存在关键点识别不全的问题")

            face, points_2d, points_3d = angle_3d[i], keypoints_2d[i], keypoints_3d[i]
            wrong_places = []

            # print("dis: " + str(arm_dis_ball(candidates[i], persons[i], balls[i])))
            # print("shoulder: " + str(shoulder(face, points_2d, points_3d, wrong_places)))
            # if arm_dis_ball(candidates[i], persons[i], balls[i]) < 3:
            #     img = images[i].get_self()
            #     draw_wrong_place(img, points_2d[8][0], points_2d[8][1])
            #     draw_wrong_place(img, points_2d[6][0], points_2d[6][1])
            #     draw_wrong_place(img, points_2d[6][0], img.shape[0])
            #     cv2.imshow(f"Detected Image, dis={arm_dis_ball(candidates[i], persons[i], balls[i])}",
            #                images[i].get_self())
            #     cv2.waitKey(0)  # 无限等待按键输入
            #     cv2.destroyAllWindows()

            if any_upper_arm_raised(images[i].get_self(), points_2d):  # 手臂抬起认为准备发球的击球
                # print("Arm raised")
                # print("ball_dis_right_arm: " + str(ball_dis_right_arm(balls[i], points_2d)))
                # print("dis: " + str(arm_dis_ball(candidates[i], persons[i], balls[i])))
                # print("shoulder: " + str(shoulder(face, points_2d, points_3d, wrong_places)))
                # cv2.imshow(f"Detected Image", images[i].get_self())
                # cv2.waitKey(0)  # 无限等待按键输入
                # cv2.destroyAllWindows()
                dis = arm_dis_ball(candidates[i], persons[i], balls[i])
                if dis < 3:  # 参数
                    detected = True
                    if balls[i][3] > points_2d[10][1]:    # 球比手低
                        wrong_places.append(points_2d[10])
                        mes.add("击球位置错误")
                    elif balls[i][3] > points_2d[11][1]:  # 球比手低
                        wrong_places.append(points_2d[11])
                        mes.add("击球位置错误")
                    # cv2.imshow(f"Detected Image, dis={dis}", images[i].get_self())
                    # cv2.waitKey(500)  # 无限等待按键输入
                    # cv2.destroyAllWindows()
                    pass
                pass

            for wrong in wrong_places:
                util.draw_wrong_place(images[i].get_self(), wrong[0], wrong[1])
            update_message(mes, total_msg, images, candidates, persons, i)
            if len(mes) > 0:
                image = images[i]
                image2 = draw_messages(image, mes)
                image2 = util.draw_bodypose(image2, candidates[i], [persons[i]])
                images[i] = image2
        if not detected:
            mes.add("未检测到发球动作")
    return list(mes)


def update_message(mes, total_msg, images, candidates, persons, i):
    if len(mes) > 0:
        image = images[i]
        image2 = draw_messages(image, mes)
        image2 = util.draw_bodypose(image2, candidates[i], [persons[i]])
        images[i].mark_wrong()
        images[i].update_self(image2)

        # cv2.imshow("Detected Image", images[i].get_self())
        # cv2.waitKey(0)  # 无限等待按键输入
        # cv2.destroyAllWindows()  # 销毁所有窗口
        total_msg.update(mes)
        mes.clear()
        Log.info(total_msg)


def shoulder(face, points_2d, points_3d, wrong_places) -> float:
    angle = 90.0
    if face > 60:  # 右肩
        angle = calculate_angle(points_2d[8], points_2d[6], points_2d[12])
        point_mid = [(points_2d[12][0] + points_2d[14][0]) / 2, (points_2d[12][1] + points_2d[14][1]) / 2]
        angle = min(angle, calculate_angle(points_2d[8], points_2d[6], point_mid))
        wrong_places.append(points_2d[6])
    elif face < -60:  # 左肩
        angle = calculate_angle(points_2d[7], points_2d[5], points_2d[11])
        point_mid = [(points_2d[11][0] + points_2d[13][0]) / 2, (points_2d[11][1] + points_2d[13][1]) / 2]
        angle = min(angle, calculate_angle(points_2d[7], points_2d[5], point_mid))
        wrong_places.append(points_2d[5])
    elif points_3d is not None:  # 正面
        angle = (calculate_angle_3d(points_3d[13], points_3d[11], points_3d[23]) + calculate_angle_3d(
            points_3d[14], points_3d[12], points_3d[24])) / 2
        wrong_places.append(points_2d[11])
        wrong_places.append(points_2d[12])
    Log.info(f"面向角度：{face}, 大臂角度{angle}")
    return angle


def calculate_angle(a, b, c):
    aa = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    bb = math.sqrt((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2)
    cc = math.sqrt((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2)
    cos_C = (aa ** 2 + bb ** 2 - cc ** 2) / (2 * aa * bb)
    angle_C_radians = math.acos(cos_C)
    angle_C_degrees = math.degrees(angle_C_radians)

    return angle_C_degrees


def calculate_angle_3d(a, b, c):
    aa = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    bb = math.sqrt((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2 + (c[2] - b[2]) ** 2)
    cc = math.sqrt((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2 + (c[2] - a[2]) ** 2)
    cos_C = (aa ** 2 + bb ** 2 - cc ** 2) / (2 * aa * bb)
    angle_C_radians = math.acos(cos_C)
    angle_C_degrees = math.degrees(angle_C_radians)

    return angle_C_degrees


def any_upper_arm_raised(img, point_2d) -> bool:
    try:
        elbow_point = point_2d[8]
        shoulder_point = point_2d[6]
        ground_point = (point_2d[6][0], img.shape[0])
        if calculate_angle(elbow_point, shoulder_point, ground_point) > 120:
            return True
    except:
        pass
    try:
        elbow_point = point_2d[7]
        shoulder_point = point_2d[5]
        ground_point = (point_2d[5][0], img.shape[0])
        if calculate_angle(elbow_point, shoulder_point, ground_point) > 120:
            return True
    except:
        return False
    return False


def ball_dis_right_arm(ball, points_2d) -> float:
    ball_point = np.array([(ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2])
    return point_dis_line(ball_point, np.array(points_2d[10]), np.array(points_2d[8]))
