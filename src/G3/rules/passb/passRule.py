import math

import cv2

from src.G3.rules.dig import armDetect, armTorsoAngle, legDetect
from src.G3.rules.passb import wristDetect
from src.utils.logger import Log
from src.utils.util import arm_dis_ball, draw_messages

from src.utils import util


def sum_rules(rawPoints, images, candidates, persons, balls, hit, hand_pose):
    mes, total_msg = set(), set()
    if len(images) == 1:  # 是图片
        # i = 0
        # image = images[i]
        # image2 = draw_messages(image, mes)
        # image2 = util.draw_bodypose(image2, candidates[i], [persons[i]])
        # images[i].mark_wrong()
        # images[i].update_self(image2)
        # mes.clear()
        # cv2.imshow("Detected Image", image.get_self())
        # cv2.waitKey(0)  # 无限等待按键输入
        # cv2.destroyAllWindows()  # 销毁所有窗口
        if candidates[0] is None or persons[0] is None:
            mes.add("人体未被识别")
            return list(mes)
        else:
            try:
                if armDetect.detect_arm_line(images[0], candidates[0], persons[0]):
                    mes.add("手臂伸太直")
                if wristDetect.detect_wrist_angle(images[0], candidates[0], persons[0], hand_pose[0]):
                    mes.add("手腕后压")
            except Exception as e:
                Log.info(hand_pose)
                mes.add("可能存在关键点识别不全的问题")
            return list(mes)
    else:
        keypoints_2d = [item["points"] for item in rawPoints]
        keypoints_3d = [item["3d_points"] for item in rawPoints]
        angle_3d = [item["3d_angle"] for item in rawPoints]
        hit_range = min(3, len(images) - hit - 1, hit - 1)
        Log.warning("解析击球之前")
        for i in range(0, hit):
            face, points_2d, points_3d = angle_3d[i], keypoints_2d[i], keypoints_3d[i]
            wrong_places = []
            if elbow(face, points_2d, points_3d, wrong_places) > 150:
                mes.add("手臂伸太直")
            else:
                wrong_places.clear()
            for wrong in wrong_places:
                util.draw_wrong_place(images[i].get_self(), wrong[0], wrong[1])
            update_message(mes, total_msg, images, candidates, persons, i)
        Log.warning("解析击球以及之后")
        for i in range(hit - hit_range, len(images)):
            if wristDetect.detect_wrist_angle(images[i], candidates[i], persons[i], hand_pose[i]):
                mes.add("手腕后压")
            update_message(mes, total_msg, images, candidates, persons, i)
        Log.info(total_msg)
    return total_msg


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


def elbow(face, points_2d, points_3d, wrong_places):
    if abs(face) > 80 or abs(face) < 30:
        angle = (calculate_angle(points_2d[6], points_2d[8], points_2d[10]) +
                 calculate_angle(points_2d[5],points_2d[7],points_2d[9])) / 2
    else:  # 正面
        angle = (calculate_angle_3d(points_3d[11], points_3d[13], points_3d[15]) + calculate_angle_3d(
            points_3d[12], points_3d[14], points_3d[16])) / 2
    wrong_places.append(points_2d[7])
    wrong_places.append(points_2d[8])
    Log.info(f"面向角度：{face}, 手肘角度{angle}")
    return angle


def shoulder(face, points_2d, points_3d, wrong_places):
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
    else:  # 正面
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
    if aa * bb < 1e-6:
        return 180
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
