import math
from numpy.linalg import svd
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    smooth_landmarks=True,
                    min_detection_confidence=0,
                    min_tracking_confidence=0)


def calculate_angle_between_three_points(point1, point2, point3):
    v1 = [(point2[i] - point1[i]) for i in range(3)]
    v2 = [(point3[i] - point2[i]) for i in range(3)]
    dot_product = sum(v1[i] * v2[i] for i in range(3))  # 计算点积
    magnitude_v1 = math.sqrt(sum(x ** 2 for x in v1))  # 计算向量的模长
    magnitude_v2 = math.sqrt(sum(x ** 2 for x in v2))
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)  # 计算夹角（弧度）
    theta = math.acos(cos_theta)
    angle_degrees = 180 - math.degrees(theta)  # 将弧度转换为角度
    return angle_degrees


def get_point(index, result):
    return [result.pose_landmarks.landmark[index].x, result.pose_landmarks.landmark[index].y,
            result.pose_landmarks.landmark[index].z]


def cal_arm_angle(result):
    point_12 = get_point(12, result)
    point_14 = get_point(14, result)
    point_16 = get_point(16, result)
    point_11 = get_point(11, result)
    point_13 = get_point(13, result)
    point_15 = get_point(15, result)
    right = calculate_angle_between_three_points(point_12, point_14, point_16)
    left = calculate_angle_between_three_points(point_11, point_13, point_15)
    return left, right


def detect_arm_angle(points):
    left = calculate_angle_between_three_points(points[11], points[13], points[15])
    right = calculate_angle_between_three_points(points[12], points[14], points[14])
    return left, right


def process_frame(img, arrays):
    def fit_plane(p1, p2, p3, p4, result):
        points = np.array([p1, p2, p3, p4])
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, v = svd(centered_points)
        normal_vector = v[-1, :]
        normal_vector[1] = 0
        normal_vector /= np.linalg.norm(normal_vector)
        if np.dot(np.array(normal_vector), np.array(get_foot(result))) < 0:
            for i in range(len(normal_vector)):
                normal_vector[i] = -normal_vector[i]
        # angle_rad = np.arctan2(normal_vector[2], normal_vector[0])
        # angle_deg = np.degrees(angle_rad) + 90
        # if angle_deg > 180:
        #     angle_deg = angle_deg - 360
        return -np.degrees(np.arctan2(-normal_vector[0], -normal_vector[2]))

    def get_foot(result):
        p1 = -np.array(get_point(27, result)) - np.array(get_point(29, result)) + 2 * np.array(get_point(31, result))
        p2 = -np.array(get_point(28, result)) - np.array(get_point(30, result)) + 2 * np.array(get_point(32, result))
        return p1 + p2

    [lefts, rights, angles, points] = arrays
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_RGB)
    if result.pose_landmarks:
        draw_spec = mp_drawing.DrawingSpec(color=(188, 188, 188), thickness=2, circle_radius=2)
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=draw_spec)
        tempPoints = []
        for i in range(32):
            tempPoints.append(get_point(i, result))
        points.append(tempPoints)
        body_angle = fit_plane(get_point(11, result), get_point(12, result), get_point(23, result),
                               get_point(24, result), result)
        lr = cal_arm_angle(result)
        rights.append(lr[1]), lefts.append(lr[0]), angles.append(body_angle)
    else:
        scaler = 1
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                          (255, 0, 0), 2)
        print('从图像中未检测出人体关键点，报错。')

    return img


def process3d(input_path):
    print('视频开始处理', input_path)
    lefts, rights, angles, points = [], [], [], []

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_path}")
        return lefts, rights, angles, points

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    print('视频总帧数为', frame_count)

    # with tqdm(total=frame_count, disable=False) as pbar:
    cnt = 1
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        try:
            process_frame(frame, [lefts, rights, angles, points])
            print(f'处理第{cnt}帧成功')
            cnt += 1
        except Exception as e:
            print('处理帧时出错:', e)
            pass
            # pbar.update(1)  # 在每一帧成功处理后更新进度条

    cap.release()
    cv2.destroyAllWindows()
    return lefts, rights, angles, points

