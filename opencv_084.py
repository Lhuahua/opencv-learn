import numpy as np
import cv2 as cv
cap = cv.VideoCapture('D:/images/video/vtest.avi')
'''
# 角点检测参数
feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3)

# KLT光流参数
lk_params = dict(winSize=(31, 31), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

# 随机颜色
color = np.random.randint(0,255,(100,3))

# 读取第一帧
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params) #shi-tomas角点检测

# 光流跟踪
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 计算光流 calcOpticalFlowPyrLK
    #   InputArray 	prevImg, // 前一帧图像
    # 	InputArray 	nextImg, // 后一帧图像
    # 	InputArray 	prevPts, // 前一帧的稀疏光流点
    # 	Size winSize = Size(21, 21), // 光流法对象窗口大小
    # 	int maxLevel = 3, // 金字塔层数，0表示只检测当前图像，不构建金字塔图像
    # 	TermCriteria criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01), // 窗口搜索时候停止条件
    #   InputOutputArray 	nextPts, // 后一帧光流点
    # 	OutputArray 	status, // 输出状态，1 表示正常该点保留，否则丢弃
    # 	OutputArray 	err, // 表示错误
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 根据状态选择
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制跟踪线
    for i, (new, old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        frame = cv.line(frame, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    cv.imshow('frame',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # 更新
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
'''
# 稠密光流分析 calcOpticalFlowFarneback
#   InputArray 	prev,前一帧
# 	InputArray 	next,后一帧
# 	InputOutputArray 	flow,光流，计算得到的移动能量场
# 	double 	pyr_scale,金字塔放缩比率
# 	int 	levels,金字塔层级数目
# 	int 	winsize,表示窗口大小
# 	int 	iterations,表示迭代次数
# 	int 	poly_n,表示光流生成时候，对邻域像素的多项式展开，n越大越模糊越稳定
# 	double 	poly_sigma,表示光流多项式展开时候用的高斯系数，n越大，sigma应该适当增加
# 	int 	flags,有两个:OPTFLOW_USE_INITIAL_FLOW表示使用盒子模糊进行初始化光流;OPTFLOW_FARNEBACK_GAUSSIAN表示使用高斯窗口
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next

cv.destroyAllWindows()
cap.release()