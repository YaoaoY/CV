import cv2
import numpy as np
# 注意！：cv2.calcOpticalFlowPyrLK() 函数返回的 next_points 值为 None 导致的。这可能是由于无法找到足够的特征点从而无法进行光流跟踪。
# 光流法参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 初始化目标位置和特征点
bbox = None
prev_points = None
prev_gray = None  # 添加 prev_gray 变量

# 创建跟踪器
tracker = cv2.TrackerCSRT_create()

# 读取视频文件
cap = cv2.VideoCapture("phoebe.mp4")

# 读取第一帧图像
ret, frame = cap.read()
if not ret:
    print("无法读取视频帧")
    exit()

# 在第一帧图像中选择目标区域
bbox = cv2.selectROI("选择目标", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# 初始化跟踪器
tracker.init(frame, bbox)

while True:
    # 读取当前帧图像
    ret, frame = cap.read()
    if not ret:
        break

    # 使用光流法跟踪特征点
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_points is None:
        # 在第一帧中计算特征点
        mask = np.zeros_like(gray)
        mask[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = 255
        prev_points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7,
                                              blockSize=7)

    if prev_points is not None:
        if prev_gray is not None:  # 添加判断 prev_gray 是否为 None 的条件
            # 计算光流
            next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

            # 选取符合条件的特征点和目标位置
            good_points = next_points[status == 1]
            x, y, w, h = cv2.boundingRect(good_points)

            # 更新目标位置和特征点
            bbox = (x, y, w, h)
            prev_points = good_points.reshape(-1, 1, 2)

            # 绘制矩形框和特征点
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for point in good_points:
                x, y = point.ravel()
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    # 显示图像
    cv2.imshow("Frame", frame)

    # 按下Esc键退出
    if cv2.waitKey(1) == 27:
        break

    # 更新前一帧图像和特征点
    prev_gray = gray.copy()

# 释放资源
cap.release()
cv2.destroyAllWindows()