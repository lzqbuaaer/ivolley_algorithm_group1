import math

import cv2

from src.G3.rules.dig import armDetect, armTorsoAngle, legDetect
from src.G3.rules.dig.ball_position import ball_position
from src.utils.logger import Log
from src.utils.util import arm_dis_ball, draw_messages

from src.utils import util


def sum_rules(rawPoints, images, candidates, persons, balls, hit):
    mes, total_msg = set(), set()
    if len(images) == 1:  # 是图片
        keypoints_2d = [item["points"] for item in rawPoints[0]]
        points_2d = keypoints_2d[0]
        wrong_places = []
        if elbow(0, points_2d, None, wrong_places) < 160:
            mes.add("手臂没有伸直")
        else:
            wrong_places.clear()
        if shoulder(0, points_2d, None, wrong_places) > 105:
            mes.add("手臂抬得太高")
        else:
            wrong_places.clear()
        for wrong in wrong_places:
            util.draw_wrong_place(images[0].get_self(), wrong[0], wrong[1])
        update_message(mes, total_msg, images, candidates, persons, 0)
    else:  # 是视频
        keypoints_2d = [item["points"] for item in rawPoints]
        keypoints_3d = [item["3d_points"] for item in rawPoints]
        angle_3d = [item["3d_angle"] for item in rawPoints]
        hit_range = min(3, len(images) - hit - 1, hit - 1)
        Log.warning("解析击球以及之前")
        for i in range(hit - hit_range, hit + hit_range):
            face, points_2d, points_3d = angle_3d[i], keypoints_2d[i], keypoints_3d[i]
            wrong_places = []
            if elbow(face, points_2d, points_3d, wrong_places)< 160:
                mes.add("手臂没有伸直")
            else:
                wrong_places.clear()
            # if i in range(hit - 1, hit + 1):
            # if not ball_position(images[i], candidates[i], persons[i], balls[i]):
            #     ball = balls[i]
            #     ball_circle = [(ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2]
            #     wrong_places.append([ball_circle[0], ball_circle[1]])
            #     mes.add("击球时球离手腕位置太远")
            for wrong in wrong_places:
                util.draw_wrong_place(images[i].get_self(), wrong[0], wrong[1])
            update_message(mes, total_msg, images, candidates, persons, i)
        Log.warning("解析击球之后，判断大臂")
        for i in range(hit + hit_range, len(images)):
            face, points_2d, points_3d = angle_3d[i], keypoints_2d[i], keypoints_3d[i]
            wrong_places = []
            if shoulder(face, points_2d, points_3d, wrong_places) > 105:
                mes.add("手臂抬得太高")
            else:
                wrong_places.clear()
            for wrong in wrong_places:
                util.draw_wrong_place(images[i].get_self(), wrong[0], wrong[1])
            update_message(mes, total_msg, images, candidates, persons, i)
        Log.warning("该动作解析完毕")
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
    angle = 180
    if face > 60:  # 右肘
        angle = calculate_angle(points_2d[6], points_2d[8], points_2d[10])
        wrong_places.append(points_2d[8])
    elif face < -60:  # 左肘
        angle = calculate_angle(points_2d[5], points_2d[7], points_2d[9])
        wrong_places.append(points_2d[7])
    elif points_3d is not None:  # 正面
        angle = (calculate_angle_3d(points_3d[11], points_3d[13], points_3d[15]) + calculate_angle_3d(
            points_3d[12], points_3d[14], points_3d[16])) / 2
        wrong_places.append(points_2d[7])
        wrong_places.append(points_2d[8])
    Log.info(f"面向角度：{face}, 手肘角度{angle}")
    return angle


def shoulder(face, points_2d, points_3d, wrong_places):
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
