import cv2 as cv
import numpy as np

src = cv.imread("test.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

#Day46 二值图像分析-连接组件标记算法
def connected_components_demo(src):
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    output = cv.connectedComponents(binary, connectivity=8, ltype=cv.CV_32S)
    num_labels = output[0]
    labels = output[1]
    colors = []
    for i in range(num_labels):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))

    colors[0] = (0, 0, 0)
    h, w = gray.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row, col] = colors[labels[row, col]]

    cv.imshow("colored labels", image)
    cv.imwrite("D:/labels.png", image)
    print("total rice : ", num_labels - 1)


#src = cv.imread("D:/images/rice.png")
h, w = src.shape[:2]
connected_components_demo(src)

#Day81-83 角点检测
# Harris角点检测 cornerHarris()，计算速度很慢
# src单通道输入图像
# blockSize计算协方差矩阵的时候邻域像素大小
# ksize表示soble算子的大小
# k表示系数
def process(image, opt=1): #将角点用圈标出
    # Detector parameters
    blockSize = 2 # 计算协方差矩阵的时候邻域像素大小
    apertureSize = 3 #表示soble算子的大小
    k = 0.04 # 系数
    # Detecting corners
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > 80:
                b = np.random.random_integers(0, 256)
                g = np.random.random_integers(0, 256)
                r = np.random.random_integers(0, 256)
                cv.circle(image, (j, i), 5, (int(b), int(g), int(r)), 2)
    # output
    return image
#src = cv.imread("D:/images/ele_panel.png")
cv.imshow("input", src)
result = process(src)
cv.imshow('result', result)
# shi-tomas角点检测，运行速度很快 goodFeaturesToTrack()
#亚像素级别角点检测cornerSubPix()
cv.waitKey(0)
cv.destroyAllWindows()


