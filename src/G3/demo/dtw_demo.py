import json
import os.path
import cv2

from matplotlib import pyplot as plt
from src.G3.demo.my_dtw import dtw
from src.media.Video import Video

triplet_indices = []

def normalize_points(points):
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    normalized_points = [((point[0] - min_x) / (max_x - min_x), (point[1] - min_y) / (max_y - min_y)) for point in
                         points]

    return normalized_points


def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_angle = dot_product / norm_product
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # 避免由于数值误差导致的无效值
    return np.degrees(angle)


import numpy as np


def calculate_center(points):
    num_points = len(points)
    if num_points == 0:
        return None
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)
    center_x = sum_x / num_points
    center_y = sum_y / num_points
    return (center_x, center_y)


def translate_points(points, reference_point):
    return [(point[0] - reference_point[0], point[1] - reference_point[1]) for point in points]


def overallSimilarity(x, y):
    # 计算每个关键点的几何中心
    x_center = calculate_center(x)
    y_center = calculate_center(y)

    x_normalized_translated = translate_points(x, x_center)
    y_normalized_translated = translate_points(y, y_center)

    # 接下来的代码与您提供的保持不变
    angles_x = [calculate_angle(x_normalized_translated[i], x_normalized_translated[j], x_normalized_translated[k]) for
                i, j, k in triplet_indices]
    angles_y = [calculate_angle(y_normalized_translated[i], y_normalized_translated[j], y_normalized_translated[k]) for
                i, j, k in triplet_indices]

    distances = [np.linalg.norm(np.array(point1) - np.array(point2)) for point1, point2 in
                 zip(x_normalized_translated, y_normalized_translated)]
    max_indexes = np.argsort(distances)[-3:]
    ret = sum(distances) + sum(angles_x) - sum(angles_y)

    return ret, max_indexes.tolist()

def main():
    video = Video('r-pass.mp4')
    standard = video.preProcess()
    video = Video('2.mp4')
    target = video.preProcess()
    with open('standard.json', 'w') as f:
        json.dump(standard, f)
    # 将 target 数据写入 JSON 文件
    with open('target.json', 'w') as f:
        json.dump(target, f)
    # standard , target = [], []
    # with open('standard.json', 'r') as f:
    #     standard = json.load(f)
    # with open('target.json', 'r') as f:
    #     target = json.load(f)
    spoints = [standard['frames'][i]['points'] for i in range(standard['actions'][0]['start'], standard['actions'][0]['end'])]
    tpoints = [target['frames'][i]['points'] for i in range(target['actions'][0]['start'], target['actions'][0]['end'])]
    spoints = [normalize_points(item) for item in spoints]
    tpoints = [normalize_points(item) for item in tpoints]
    d, cost_matrix, acc_cost_matrix, path, bads = dtw(spoints, tpoints, dist=overallSimilarity)
    print(bads)
    # plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
    # plt.plot(path[0], path[1], 'w')
    # plt.show()
    for i in range(len(path[0])):
        # draw_bodypose
        plt.figure(figsize=(10, 6))
        plt.gca().invert_yaxis()
        plt.scatter(*zip(*spoints[path[0][i]]), label=f'Standard Points {i + 1}')
        plt.scatter(*zip(*tpoints[path[1][i]]), label=f'Sample Points {i + 1}')
        plt.legend()

        # 添加预定义的连线
        lines_set = [
            [(6, 8), (8, 10), (6, 12), (12, 14), (14, 16)],  # 线组1
            [(12, 11), (5, 6), (0, 5), (0, 6), (0, 3), (0, 4), (1, 3), (2, 4)],  # 线组2
            [(5, 7), (7, 9), (5, 11), (11, 13), (13, 15)]  # 线组3
        ]
        colors = ['red', 'blue', 'green']  # 线组颜色

        for lines, color in zip(lines_set, colors):
            for line in lines:
                plt.plot(*zip(*[spoints[path[0][i]][point] for point in line]), color=color)
                plt.plot(*zip(*[tpoints[path[1][i]][point] for point in line]), color=color,linestyle='dotted')  # 添加样本点的连线，使用虚线

        plt.legend()
        plt.xlim(-0.7 , 1.8)
        plt.savefig(f'tmp_frame_{i}.png')
        plt.close()

    frames = []
    for i in range(len(path[0])):
        for _ in range(10):
            img = cv2.imread(f'tmp_frame_{i}.png')
            frames.append(img)

    height, width, _ = frames[0].shape
    video_path = 'output_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()


    for i in range(len(path[0])):
        file_path = f'tmp_frame_{i}.png'
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    main()
