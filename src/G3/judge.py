import numpy as np

from src.G3.rules.rule import Rule
from src.enums.action import Action
from src.media.Video import NEEDED_FLAG

from src.utils.logger import Log

rule_dig = Rule(Action.Dig)
rule_pass = Rule(Action.Pass)
rule_serve = Rule(Action.Serve)


def judge_video(type, datas, video):
    actions = datas['actions']
    frames = datas['frames']

    msg = set()
    serve_detected = False
    for i, action in enumerate(actions):  # 一个片段一个片段处理
        curAction = frames[action['start']:action['end']]
        Log.warning(f"开始解析动作：{action['start']} {action['hit']} {action['end']}, ")
        candidates = []
        persons = []
        balls = []
        for item in curAction:
            candidate = [point + [j] for j, point in enumerate(item['points'])]
            candidates.append(np.array(candidate))
            person = np.array([0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 1.0, 17])
            persons.append(person)
            balls.append(item['ball'])

        images = video.frames[action['start']:action['end']]

        if type[i] == 0:  # Action.Dig:
            msg_action = rule_dig(curAction, images, candidates, persons, balls, hit=(action['hit'] - action['start']))
        elif type[i] == 4:  # Action.Pass:
            hand_pose = []
            for item in curAction:
                hand_pose.append(item['hands'])
            msg_action = rule_pass(curAction, images, candidates, persons, balls, hit=(action['hit'] - action['start']),
                                   hand_pose=hand_pose)
        elif type[i] == 3:  # Action.Serve:
            msg_action = rule_serve(curAction, images, candidates, persons, balls)
            if "未检测到发球动作" not in msg_action:
                serve_detected = True
        else:
            return msg, None
        msg.update(msg_action)
        Log.info(f"action: {i} : {msg_action}")
    output_path = video.close()
    if serve_detected:
        try:
            msg.remove("未检测到发球动作")
        except KeyError:
            pass
    Log.info(msg)
    return msg, output_path


def judge_image(type, face, image, src_path):
    msg = set()
    keyPoints = []
    ball = image.mark_ball()
    points, scores, hands, hands_scores = image.mark_wholebody()
    Log.info(f"{len(points)} != 0 and {len(scores)} != 0 and {len(hands_scores)} != 0 and {len(hands)} != 0")
    if len(points) != 0 and len(scores) != 0 and len(hands_scores) != 0 and len(hands) != 0:
        keyPoints.append(
            {"flag": NEEDED_FLAG, "points": points, "scores": scores, "ball": image.get_ball(), "hands": hands,
             "hands_scores": hands_scores})
    else:
        msg.add("未检测到人体")
        return msg, src_path
    candidates = []
    persons = []
    balls = []
    for item in keyPoints:
        Log.info(f"item{item}")
        candidate = [point + [j] for j, point in enumerate(item['points'])]
        candidates.append(np.array(candidate))
        person = np.array([0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 1.0, 17])
        persons.append(person)
        balls.append(item['ball'])
        Log.info(f"tag in judge {type}")
        if type == 0:  # Action.Dig:
            msg_action = rule_dig([keyPoints], [image], candidates, persons, balls)
        elif type == 4:  # Action.Pass:
            hand_pose = []
            hand_pose.append(item['hands'])
            msg_action = rule_pass([keyPoints], [image], candidates, persons, balls, hand_pose=hand_pose)
        elif type == 3:  # Action.Serve:
            msg_action = rule_serve([keyPoints], [image], candidates, persons, balls)
        else:
            return msg, None
        msg.update(msg_action)

    output_path = image.close()
    Log.info(msg)
    return msg, output_path
