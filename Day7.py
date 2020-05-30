import cv2 as cv
import numpy as np

#视频处理
capture = cv.VideoCapture("D:/images/video/test.mp4") #视频读取与解码，支持各种视频格式、网络视频流、摄像头读取
height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
count = capture.get(cv.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv.CAP_PROP_FPS)
print(height, width, count, fps)

#Day77 视频读写与处理
def process(image, opt=1): #处理操作，0做取反，1高斯模糊2边缘提取
    dst = None
    if opt == 0:
        dst = cv.bitwise_not(image)
    if opt == 1:
        dst = cv.GaussianBlur(image, (0, 0), 15)
    if opt == 2:
        dst = cv.Canny(image, 100, 200)
    return dst
index = 0
while(True):
    ret, frame = capture.read() #读取一帧图像
    if ret is True:
        cv.imshow("video-input", frame)
        c = cv.waitKey(50)  #每个50ms刷新一次
        if c >= 49: #49代表1
            index = c -49
        result = process(frame, index)
        cv.imshow("result", result)
        print(c)
        if c == 27:  #ESC
            break
    else:
        break

#Day78 识别与跟踪视频中的特定颜色对象
# 色彩转换BGR2HSV
# inRange提取颜色区域mask
# 对mask区域进行二值分析得到位置与轮廓信息
# 绘制外接椭圆与中心位置
# 显示结果
def process(image, opt=1):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    line = cv.getStructuringElement(cv.MORPH_RECT, (15, 15), (-1, -1))
    mask = cv.inRange(hsv, (0, 43, 46), (10, 255, 255))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, line)
    # 轮廓提取, 发现最大轮廓
    out, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    index = -1
    max = 0
    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        if area > max:
            max = area
            index = c
    # 绘制
    if index >= 0:
        rect = cv.minAreaRect(contours[index])
        cv.ellipse(image, rect, (0, 255, 0), 2, 8)
        cv.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
    return image
while(True):
    ret, frame = capture.read()
    if ret is True:
        cv.imshow("video-input", frame)
        result = process(frame)
        cv.imshow("result", result)
        c = cv.waitKey(50)
        print(c)
        if c == 27:  #ESC
            break
    else:
        break
#Day79 背景/前景 提取：背景消除技术，通过对前面一系列帧提取背景模型，与当前帧进行相减。一种是基于高斯混合模型GMM实现的背景提取，另外一种是基于最近邻KNN实现的。
# history表示过往帧数，500帧，选择history = 1就变成两帧差
# varThreshold表示像素与模型之间的马氏距离，值越大，只有那些最新的像素会被归到前景，值越小前景对光照越敏感。
# detectShadows 是否保留阴影检测，请选择False这样速度快点
cap = cv.VideoCapture('D:/images/video/color_object.mp4')
fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=1000, detectShadows=False)
#cv.createBackgroundSubtractorKNN
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    background = fgbg.getBackgroundImage()
    cv.imshow('input', frame)
    cv.imshow('mask',fgmask)
    cv.imshow('background', background)
    k = cv.waitKey(10)&0xff
    if k == 27:
        break
#Day80 背景消除与前景ROI提取:现对前景活动对象ROI区域的提取,是很多视频监控分析软件常用的手段之一
# 1.初始化背景建模对象GMM
# 2.读取视频一帧
# 3.使用背景建模消除生成mask
# 4.对mask进行轮廓分析提取ROI
# 5.绘制ROI对象
cap = cv.VideoCapture('D:/images/video/vtest.avi')
fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
def process(image, opt=1):
    mask = fgbg.apply(frame)
    line = cv.getStructuringElement(cv.MORPH_RECT, (1, 5), (-1, -1))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, line)
    cv.imshow("mask", mask)
    # 轮廓提取, 发现最大轮廓
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        if area < 100:
            continue
        rect = cv.minAreaRect(contours[c])
        cv.ellipse(image, rect, (0, 255, 0), 2, 8)
        cv.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
    return image, mask

while True:
    ret, frame = cap.read()
    cv.imwrite("D:/input.png", frame)
    cv.imshow('input', frame)
    result, m_ = process(frame)
    cv.imshow('result', result)
    k = cv.waitKey(50)&0xff
    if k == 27:
        cv.imwrite("D:/result.png", result)
        cv.imwrite("D:/mask.png", m_)
        break
#Day81-83 角点检测见Day5文件
#Day84-85 移动对象的KLT光流跟踪算法,是稀疏光流跟踪算法 calcOpticalFlowPyrLK
#Day86  稠密光流分析：基于前后两帧所有像素点的移动估算算法，其效果要比稀疏光流算法更好 calcOpticalFlowFarneback
#Day87-90 移动对象分析，移动轨迹绘制
cap.release()
cv.destroyAllWindows()

