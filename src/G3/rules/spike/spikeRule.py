import math
import cv2

from src.utils.logger import Log
from src.utils.util import draw_messages
from src.utils import util

def spike_analysis(rawPoints, images, candidates, persons, balls, hit):
    mes, total_msg = set(), set()
    if len(images) == 1:  # 是图片
        keypoints_2d = [item["points"] for item in rawPoints[0]]
        points_2d = keypoints_2d[0]
        wrong_places = []
        Log.warning("解析扣球动作")
        
        if arm_preparation_hitting_hand(0, points_2d, None, wrong_places) < 90:
            mes.add("击球手肘没有伸直")
        if spike_swing(face, points_2d, points_3d, wrong_places) < 150:
            mes.add("手肘未伸直")
        for wrong in wrong_places:
            util.draw_wrong_place(images[0].get_self(), wrong[0], wrong[1])
            update_message(mes, total_msg, images, candidates, persons, 0)
        
    else:  # 是视频
        keypoints_2d = [item["points"] for item in rawPoints]
        keypoints_3d = [item["3d_points"] for item in rawPoints]
        angle_3d = [item["3d_angle"] for item in rawPoints]
        hit_range = min(1, len(images) - hit - 1, hit - 1)
        wrong_places = []
        
        Log.warning("解析扣球动作")
        
        for i in range(0, hit - hit_range):
            # 引臂时击球手肘要高于肩膀
            face, points_2d, points_3d = angle_3d[i], keypoints_2d[i], keypoints_3d[i]
            wrong_places = []

            if arm_preparation_hitting_hand(face, points_2d, points_3d, wrong_places) < 90:
                mes.add("击球手肘没有伸直")
            else:
                wrong_places.clear()
                break
        for wrong in wrong_places:
            util.draw_wrong_place(images[i].get_self(), wrong[0], wrong[1])
        update_message(mes, total_msg, images, candidates, persons, i)
    
        for i in range(hit - hit_range, hit + hit_range):
            # 击球时手臂要伸直
            face, points_2d, points_3d = angle_3d[i], keypoints_2d[i], keypoints_3d[i]
            wrong_places = []

            if arm_preparation_hitting_hand(face, points_2d, points_3d, wrong_places) < 150:
                mes.add("手臂未伸直")
            else:
                wrong_places.clear()
            
            for wrong in wrong_places:
                util.draw_wrong_place(images[i].get_self(), wrong[0], wrong[1])
            update_message(mes, total_msg, images, candidates, persons, i)

        for i in range(hit - hit_range, hit + hit_range):
            # 击球时手肘要伸直
            face, points_2d, points_3d = angle_3d[i], keypoints_2d[i], keypoints_3d[i]
            wrong_places = []

            if spike_swing(face, points_2d, points_3d, wrong_places) < 150:
                mes.add("手肘未伸直")
            else:
                wrong_places.clear()
            
            for wrong in wrong_places:
                util.draw_wrong_place(images[i].get_self(), wrong[0], wrong[1])
            update_message(mes, total_msg, images, candidates, persons, i)
            
    Log.warning("扣球动作解析完毕")
    return total_msg


def update_message(mes, total_msg, images, candidates, persons, i):
    if len(mes) > 0:
        image = images[i]
        image2 = draw_messages(image, mes)
        image2 = util.draw_bodypose(image2, candidates[i], [persons[i]])
        images[i].mark_wrong()
        images[i].update_self(image2)
        total_msg.update(mes)
        mes.clear()
        Log.info(total_msg)


def arm_preparation_hitting_hand(face, points_2d, points_3d, wrong_places):
    angle = 90.0
    if face > 60 or face < -60:  # 侧面
        angle = calculate_angle(points_2d[8], points_2d[6], points_2d[12])
        wrong_places.append(points_2d[6])
    elif points_3d is not None:  # 正面
        angle = calculate_angle_3d(points_3d[14], points_3d[12], points_3d[24])
        wrong_places.append(points_2d[6])
    Log.info(f"面向角度：{face}, 大臂角度{angle}")
    return angle


def spike_swing(face, points_2d, points_3d, wrong_places):
    angle = 180
    if face > 60 or face < -60:  # 侧面
        angle = calculate_angle(points_2d[6], points_2d[8], points_2d[10])
        wrong_places.append(points_2d[8])
    elif points_3d is not None:  # 正面
        angle = calculate_angle_3d(points_3d[12], points_3d[14], points_3d[16])
        wrong_places.append(points_2d[8])
    Log.info(f"面向角度：{face}, 手肘角度{angle}")
    return angle


def calculate_angle(a, b, c):
    aa = math.dist(a, b)
    bb = math.dist(c, b)
    cc = math.dist(c, a)
    cos_C = (aa ** 2 + bb ** 2 - cc ** 2) / (2 * aa * bb)
    angle_C_radians = math.acos(cos_C)
    angle_C_degrees = math.degrees(angle_C_radians)
    return angle_C_degrees


def calculate_angle_3d(a, b, c):
    aa = math.dist(a, b)
    bb = math.dist(c, b)
    cc = math.dist(c, a)
    cos_C = (aa ** 2 + bb ** 2 - cc ** 2) / (2 * aa * bb)
    angle_C_radians = math.acos(cos_C)
    angle_C_degrees = math.degrees(angle_C_radians)
    return angle_C_degrees
