import cv2 as cv

capture = cv.VideoCapture(0)

# HAAR级联检测器使用：支持人脸检测、微笑、眼睛与嘴巴检测等，通过加载这些预先训练的HAAR模型数据可以实现相关的对象检测，
detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")
while True:
    ret, image = capture.read()
    if ret is True:
        cv.imshow("frame", image)
        faces = detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=1,
                                          minSize=(30, 30), maxSize=(120, 120))
        for x, y, width, height in faces:
            cv.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2, cv.LINE_8, 0)
        cv.imshow("faces", image)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break
# HAAR特征介绍
# HAAR小波基函数，因为其满足对称性，对人脸这种生物对称性良好的对象特别适合用来做检测器，常见的Haar特征分为三类：
# 边缘特征、
# 线性特征、
# 中心特征和对角线特征，
# 不同特征可以进行多种组合，生成更加复杂的级联特征，特征模板内有白色和黑色两种矩形，并定义该模板的特征值为白色矩形像素和减去黑色矩形像素和，Haar特征值反映了图像的对比度与梯度变化。
# OpenCV中HAAR特征计算是积分图技术，这个我们在前面也分享过啦，所以可以非常快速高效的开窗检测， HAAR级联检测器具备有如下特性：
# -	高类间变异性
# -	低类内变异性
# -	局部强度差
# -	不同尺度
# -	计算效率高
capture = cv.VideoCapture(0)
face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")
smile_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")
while True:
    ret, image = capture.read()
    if ret is True:
        cv.imshow("frame", image)
        faces = face_detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=3,
                                          minSize=(30, 30), maxSize=(300, 300))
        for x, y, width, height in faces:
            cv.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2, cv.LINE_8, 0)
        roi = image[y:y+height,x:x+width]
        smiles = smile_detector.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=3,
                                               minSize=(15, 15), maxSize=(100, 100))
        for sx, sy, sw, sh in smiles:
            cv.rectangle(roi, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)

        cv.imshow("faces", image)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break
# 对象检测-LBP特征介绍
# 局部二值模式(Local Binary Pattern)主要用来实现2D图像纹理分析。其基本思想是用每个像素跟它周围的像素相比较得到局部图像结构，假设中心像素值大于相邻像素值则则相邻像素点赋值为1，否则赋值为0，最终对每个像素点都会得到一个二进制八位的表示，比如11100111。假设3x3的窗口大小，这样对每个像素点来说组合得到的像素值的空间为[0~2^8]。这种结果称为图像的局部二值模式或者简写为了LBP。
capture = cv.VideoCapture(0)
detector = cv.CascadeClassifier("D:/opencv-4.0.0/opencv/build/etc/lbpcascades/lbpcascade_frontalface_improved.xml")
while True:
    ret, image = capture.read()
    if ret is True:
        cv.imshow("frame", image)
        faces = detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=1,
                                          minSize=(30, 30), maxSize=(120, 120))
        for x, y, width, height in faces:
            cv.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2, cv.LINE_8, 0)
        cv.imshow("faces", image)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break

cv.destroyAllWindows()

