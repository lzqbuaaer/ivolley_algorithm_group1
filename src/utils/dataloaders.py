import copy
from pathlib import Path

import cv2

from src.utils.detect import detect_person, detect_ball, ROOT, detect_wholebody
from src.utils.logger import Log
from src.utils.util import arm_dis_ball


class VideoLoader:
    output_num = "002"
    origin_format = output_num + ".avi"
    output_format = output_num + ".mp4"

    # 初始化后定位到指定动作开始位置，注意重写_satisfy方法，不可直接调用此父类
    def __init__(self, url):
        self.gap = 2
        self.cnt = 0  # 原视频当前处理帧
        self.frames = dict()  # 视频的每一帧
        self.detect_num = []  # 选取了哪些帧号的帧用于检测
        self.p = 0  # detect_num的指针
        self.videoCapture = cv2.VideoCapture(url)  # 要解析的视频对象
        self.fps = int(self.videoCapture.get(cv2.CAP_PROP_FPS))  # 原视频帧率
        self.output_path = str(ROOT) + "/output/" + Path(url).stem
        self.video = cv2.VideoWriter(self.output_path + VideoLoader.output_format,
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     self.fps,
                                     (int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        Log.info("视频帧率:%d" % self.fps)

        loop_cnt = 0
        while True:
            # 隔gap次帧进行检测，写法受限于self.videoCapture.read()
            while loop_cnt < self.gap:
                success, frame = self.videoCapture.read()
                if not success:
                    self.close()
                    raise StopIteration
                # 开始的帧都视为未检测的
                loop_cnt += 1
                self.cnt += 1
                self.add_frame(frame)  # 默认detect为false，此时将当前帧直接按cnt放入frames中
            # 对当前帧frame进行检测
            Log.debug("第%d帧检测开始" % self.cnt)
            candidate, person = detect_person(frame)
            ball = detect_ball(frame)
            self.set_gap(candidate, person, ball)  # 根据手球距离调整gap，不理解
            loop_cnt = 0
            try:
                # 各子类自行重写的方法
                if self._satisfy(candidate, person, ball):
                    break
            except Exception as e:
                # traceback.print_exc()
                Log.error("第%d帧" % self.cnt + str(e))
            # cv2.imshow("ii", frame)
            # cv2.waitKey(0)
        self.round = 1
        self.sustaining = False
        # Log.debug("视频已定位到垫球动作开始位置，从第%d帧开始检测" % self.cnt)
        Log.debug("视频已定位到动作开始位置，从第%d帧开始检测" % self.cnt)

    # 本方法用于从动作开始位置开始检测
    def get_all_pic(self):
        candidates = []
        persons = []
        balls = []
        frames = []

        # 注意本类实现了__iter__和__next__方法，使得如下写法可行
        for candidate, person, ball, frame in self:
            candidates.append(candidate)
            persons.append(person)
            balls.append(ball)
            frames.append(frame)
            # if self.round > 1:#todo 判断条件改为手触球
            #    break

        # 用于写入视频初始化
        self.detect_num.append(0)
        return candidates, persons, balls, frames

    def add_frame(self, frame, marks=None, detect=False):
        if marks is None:
            marks = []
        if detect:
            print("检测" + str(self.detect_num[self.p]))
            num = self.detect_num[self.p - 1] + 1
            length = 1
            print(marks[self.p])
            if marks[self.p] == 1:
                length = int(self.fps / 2)
            while num < self.detect_num[self.p]:
                for j in range(length):
                    self.video.write(self.frames[num])
                    print("写入" + str(num))
                num += 1
            self.p += 1
            for i in range(length):
                print("写入")
                self.video.write(frame)
        else:
            # self.video.write(frame)
            # 将帧先存起来，后续完成需要检测的帧后再将前面的帧一同写入
            self.frames[self.cnt] = frame

    def set_gap(self, candidate, person, ball):
        try:
            dis = arm_dis_ball(candidate, person, ball)
            Log.info("球离手的距离是球的半径的%f倍" % dis)
            if dis < 5:
                self.gap = 1
            elif dis < 9:
                self.gap = 3
            else:
                self.gap = 5
        except Exception as e:
            Log.error("第%d帧" % self.cnt + str(e))

    def close(self):
        self.video.release()
        Log.info("结果视频已生成")

        # 将cv2导出的avi格式转成mp4格式
        # os.system(f'ffmpeg -i "{self.output_path + self.origin_format}" -vcodec h264 "{self.output_path + self.output_format}"')

    def _satisfy(self, candidate, person, ball):
        return False

    def test(self):
        for i in range(len(self.frames)):
            cv2.imshow("ii", self.frames[i + 1])
            cv2.waitKey(0)

    def __iter__(self):
        # self.cnt = 0
        return self

    def __next__(self):
        loop_cnt = 0
        while True:
            success, frame = self.videoCapture.read()
            if not success:
                raise StopIteration

            loop_cnt += 1
            self.cnt += 1
            self.add_frame(frame)
            if loop_cnt >= self.gap:
                break

        Log.info("第%d帧检测开始" % self.cnt)
        candidate, person = detect_person(frame)
        ball = detect_ball(frame)
        self.set_gap(candidate, person, ball)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        # 该函数处于动作识别过程中，若未检测到关键点应该直接将该异常抛给更高层
        try:
            result = self._satisfy(candidate, person, ball)  # 球手距离不能太远
        except Exception as e:
            Log.error("第%d帧存在关键点无法检测的行为" % self.cnt)
            return self.__next__()

        # 0. 第一次迭代的self.cnt 满足 self._satisfy，result == true, sustaining = false
        # 2. 触发条件：上一帧没有检测到动作（sustaining == true），这一帧检测到了（result == true）
        if result and self.sustaining:
            self.round += 1
            # 3. round == 2，get_all_pic方法跳出for循环
            self.sustaining = False

        # 1. 如果没有检测到动作，则需要继续检测（sustaining == 1）
        # todo 改成可以检测多次垫球
        self.sustaining = not result

        self.detect_num.append(self.cnt)

        return candidate, person, ball, frame


class DigVideoLoader(VideoLoader):
    def __init__(self, url):
        super().__init__(url)
        self.gap = 1

    # 1. 球位于小臂之间
    # 2. 球与手臂保持水平
    def _satisfy(self, candidate, person, ball):
        # dis = arm_dis_ball(candidate, person, ball)  # 球心到小臂距离和球半径的比值
        # return dis < 2
        return True

    def get_all_pic(self):
        candidates = []
        persons = []
        balls = []
        frames = []
        while True:
            success, frame = self.videoCapture.read()
            if not success:
                # raise StopIteration
                break
            else:
                Log.info("第%d帧检测开始" % self.cnt)
                self.cnt += 1
                self.detect_num.append(self.cnt)
                self.add_frame(frame)

                candidate, person = detect_person(frame)
                ball = detect_ball(frame)
                candidates.append(candidate)
                persons.append(person)
                balls.append(ball)
                frames.append(frame)

        return candidates, persons, balls, frames

    # def ball_test(self):
    #     balls = []
    #     frames = []
    #     while True:
    #         success, frame = self.videoCapture.read()
    #         if not success:
    #             break
    #         else:
    #             Log.info("第%d帧检测开始" % self.cnt)
    #             self.cnt += 1
    #             self.detect_num.append(self.cnt)
    #             self.add_frame(frame)
    #
    #             ball = detect_ball(frame)
    #             tmpImg = copy.deepcopy(frame)
    #             if ball is not None:
    #                 cv2.rectangle(tmpImg, (int(ball[0]), int(ball[1])), (int(ball[2]), int(ball[3])), (255, 0, 255), 2)
    #             cv2.imshow("tmp", tmpImg)
    #             cv2.waitKey(50)
    #             balls.append(ball)
    #             frames.append(frame)


class BallVideoLoader(VideoLoader):
    output_num = "_ball"

    def __init__(self, url):
        super().__init__(url)
        self.gap = 1
        self.video.release()
        self.video = cv2.VideoWriter(self.output_path + "_ball.mp4",
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     self.fps,
                                     (int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # 1. 球位于小臂之间
    # 2. 球与手臂保持水平
    def _satisfy(self, candidate, person, ball):
        # dis = arm_dis_ball(candidate, person, ball)  # 球心到小臂距离和球半径的比值
        # return dis < 2
        return True

    def get_all_pic(self):
        candidates = []
        persons = []
        balls = []
        frames = []
        while True:
            success, frame = self.videoCapture.read()
            if not success:
                # raise StopIteration
                break
            else:
                Log.info("第%d帧检测开始" % self.cnt)
                self.cnt += 1
                self.detect_num.append(self.cnt)
                self.add_frame(frame)

                candidate, person = detect_person(frame)
                ball = detect_ball(frame)
                candidates.append(candidate)
                persons.append(person)
                balls.append(ball)
                frames.append(frame)

        return candidates, persons, balls, frames

    def ball_test(self):
        balls = []
        frames = []
        while True:
            success, frame = self.videoCapture.read()
            if not success:
                break
            else:
                Log.info("第%d帧检测开始" % self.cnt)
                self.cnt += 1
                # self.detect_num.append(self.cnt)
                # self.add_frame(frame)

                ball = detect_ball(frame)
                # tmpImg = copy.deepcopy(frame)
                if ball is not None:
                    self.video.write(frame)
                    # cv2.rectangle(tmpImg, (int(ball[0]), int(ball[1])), (int(ball[2]), int(ball[3])), (255, 0, 255), 2)
                # cv2.imshow("tmp", tmpImg)
                # cv2.waitKey(50)
                # balls.append(ball)
                # frames.append(frame)
        self.close()

    def wholebody_test(self):
        frames = []
        while True:
            success, frame = self.videoCapture.read()
            if not success:
                break
            else:
                Log.info("第%d帧检测开始" % self.cnt)
                self.cnt += 1
                # self.detect_num.append(self.cnt)
                # self.add_frame(frame)
                wholebodys = detect_wholebody(frame)
                # tmpImg = copy.deepcopy(frame)
                print(wholebodys['predictions'][0])
                ball = detect_ball(frame)
                if ball is not None:
                    self.video.write(frame)
                    # cv2.rectangle(tmpImg, (int(ball[0]), int(ball[1])), (int(ball[2]), int(ball[3])), (255, 0, 255), 2)
                # cv2.imshow("tmp", tmpImg)
                # cv2.waitKey(50)
                # balls.append(ball)
                # frames.append(frame)
        self.video.release()
        self.close()



if __name__ == "__main__":
    test = BallVideoLoader("../../videos/wrong.mp4")
    test.wholebody_test()
