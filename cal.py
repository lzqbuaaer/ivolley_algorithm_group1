import json
import matplotlib.pyplot as plt
file_path = '/predictions/video_20240123_134915.json'
file_path = '/predictions/video_20240123_133925.json'
file_path = '/predictions/video_20240123_133930.json'

# 读取JSON文件
with open(file_path, 'r') as f:
    data = json.load(f)

# 初始化字典来存储每个关键点的得分总和和帧数
keypoint_scores_sum = {i: 0 for i in range(133)}
keypoint_frame_count = {i: 0 for i in range(133)}

# 遍历每一帧
for frame_data in data:
    frame_data = frame_data['instances'][0]
    keypoint_scores = frame_data['keypoint_scores']  # 获取当前帧的关键点得分列表

    # 累加每个关键点的得分总和
    for idx, score in enumerate(keypoint_scores):
        keypoint_scores_sum[idx] += score
        keypoint_frame_count[idx] += 1

# 计算每个关键点的得分平均值
keypoint_scores_avg = {}
for idx, count in keypoint_frame_count.items():
    if count != 0:
        keypoint_scores_avg[idx] = keypoint_scores_sum[idx] / count
    else:
        keypoint_scores_avg[idx] = 0

# 可视化关键点的平均得分
plt.figure(figsize=(12, 6))
plt.plot(keypoint_scores_avg.keys(), keypoint_scores_avg.values(), marker='o', linestyle='-')
plt.title('Average Keypoint Scores Over Frames')
plt.xlabel('Keypoint Index')
plt.ylabel('Average Score')
plt.grid(True)
plt.tight_layout()
plt.show()
