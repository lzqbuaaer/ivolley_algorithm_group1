"""
检测手臂与躯干最大角度不超过110°
"""
import math

import cv2
from settings import *
from src.utils import util
from src.utils.logger import Log


def judge_angle(xm, ym, x1, y1, x2, y2) -> bool:
    a = math.sqrt(math.pow((xm - x1), 2) + math.pow((ym - y1), 2))
    b = math.sqrt(math.pow((xm - x2), 2) + math.pow((ym - y2), 2))
    c = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

    angle = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))

    Log.debug("手臂与躯干的角度为%f" % angle)
    # print("angle of arm and torso is: %f" % angle)
    return angle <= 130


def detect(image, candidate, person) -> bool:
    flag = True

    # print(candidate[person[5]][0:2])
    # 判断左臂与躯干，选取5,6,11
    try:
        img = image.get_self()
        if not judge_angle(*util.num2pos(5, candidate, person),
                           *util.num2pos(6, candidate, person),
                           *util.num2pos(11, candidate, person)):
            flag = False

            util.draw_wrong_place(img, *util.num2pos(5, candidate, person))
            image.update_self(img)

            Log.info("左臂抬起过高")

    except Exception as e:
        Log.error(e)
    # 判断右臂与躯干，选取2,3,8
    try:
        img = image.get_self()
        if not judge_angle(*util.num2pos(2, candidate, person),
                           *util.num2pos(3, candidate, person),
                           *util.num2pos(8, candidate, person)):

            flag = False
            util.draw_wrong_place(img, *util.num2pos(2, candidate, person))
            image.update_self(img)

            Log.info("右臂抬起过高")

    except Exception as e:
        Log.error(e)
    Log.debug("armTorsoAngle执行完成")

    return flag
