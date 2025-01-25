import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt

# 生成随时间变化的多维序列
seq1 = [1, 3, 3, 7, 10, 14, 16, 18]
seq2 = [1, 3, 5, 7, 9, 11, 13]

# 定义两个序列之间的距离度量函数
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# 计算 DTW
d, cost_matrix, acc_cost_matrix, path = dtw(seq1, seq2, dist=euclidean_distance)

# 提取匹配路径
match_indices_seq1 = path[0]
match_indices_seq2 = path[1]

# 假设您想要比较第3个点的相似性
point_index = 2  # 由于 Python 中索引从 0 开始，因此第3个点的索引为2

# 如果第3个点在匹配路径中，表示该点在两个序列中具有相似的对应点
if point_index in match_indices_seq1 and point_index in match_indices_seq2:
    similarity_index = np.where(match_indices_seq1 == point_index)[0][0]
    print("Point at index", point_index, "is matched between the sequences.")
    print("Similarity index in seq2:", match_indices_seq2[similarity_index])
else:
    print("Point at index", point_index, "is not matched between the sequences.")
