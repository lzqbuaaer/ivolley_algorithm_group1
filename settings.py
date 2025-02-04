# path
import os

ImageOutputPath = os.path.join("output", "algorithm_2_img")
VideoOutputPath = os.path.join("output", "algorithm_2_video")
AlgorithmPath = os.path.dirname(os.path.abspath(__file__))
print(f"AlgorithmPath = {AlgorithmPath}")
PythonPath = 'D:\Anaconda\\envs\\volley\\Lib\\site-packages'
# PythonPath = '/usr/local/lib/python3.10/dist-packages'
pose2dPath = os.path.join('mmpose', '.mim', 'configs', 'wholebody_2d_keypoint', 'topdown_heatmap',
                          'coco-wholebody', 'td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py')

tag_dict = {
    0: '垫球',
    4: '传球',
    3: '发球',
    1: '扣球',
    2: '拦网',
    5: '其它'
}

MIN_BBOX_SCORE = 0.7
MIN_POINT_SCORE = 0.1

CORRECT_MARK = 1
WRONG_MARK = 15
