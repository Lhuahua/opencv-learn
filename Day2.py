import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



src = cv.imread("test.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
'''
#Day11 像素归一化：NORM_L1和为1（L1-范数）,NORM_L2向量模为1（L2-范数）,NORM_INF最大值（L∞范数）,NORM_MINMAX减去最小值再除以该值
# 转换为浮点数类型数组
gray = np.float32(gray)
print(gray)
# scale and shift by NORM_MINMAX
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst=dst, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX) #beta只在MINMAX上起作用
print(dst)
cv.imshow("NORM_MINMAX", np.uint8(dst*255))
# scale and shift by NORM_INF
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_INF)
print(dst)
cv.imshow("NORM_INF", np.uint8(dst*255))
# scale and shift by NORM_L1
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_L1)
print(dst)
cv.imshow("NORM_L1", np.uint8(dst*10000000)) #因为归一化的数值太小（小于1）， 需要放大到0-255范围
# scale and shift by NORM_L2
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_L2)
print(dst)
cv.imshow("NORM_L2", np.uint8(dst*10000))

#Day13 图像翻转
dst = cv.flip(src,0) #0:X轴翻转，1Y轴翻转，2对角翻转

#Day14 图像插值 cv.resize()四种插值法：双立方插值，双线性内插值，Lanczos采样放缩算法
h, w = src.shape[:2]
dst = cv.resize(src, (w*2, h*2), fx=0.75, fy=0.75, interpolation=cv.INTER_NEAREST) #INTER_NEAREST,INTER_LINEAR ,INTER_CUBIC ,INTER_LANCZOS4
dst = cv.resize(src, (w*2, h*2), interpolation=cv.INTER_LINEAR)

#Day15 几何图形绘制

#Day16 ROI：range of interest
h,w = src.shape[:2]
cy = h/2
cx = w/2
roi = src[cy-100:cy+100,cx-100:cx+100,:] #规则ROI
image= np.copy(roi)
mask = cv.inRange(src,(),()) #不规则ROI需要mask，与原图位操作

#Day17 图像直方图：灰度图像的阈值分割、基于颜色的图像检索以及图像分类、反向投影跟踪。有灰度直方图和颜色直方图
#cv.calcHist([images],[i],None,[256],[0,256])
def image_hist(image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
plt.hist(gray.ravel(),256,[0,256])  #灰度直方图
plt.show()
image_hist(src) #颜色直方图

#Day18 图像直方图均衡化，用于图像增强、对输入图像进行直方图均衡化处理，提升后续对象检测的准确率。在OpenCV人脸检测的代码演示中已经很常见。此外对医学影像图像与卫星遥感图像也经常通过直方图均衡化来提升图像质量。
dst= cv.equalizeHist(gray)
cv.imshow("",dst)
plt.hist(gray.ravel(),256,[0,256])  #灰度直方图
plt.show()
plt.hist(dst.ravel(),256,[0,256])  #灰度直方图
plt.show()

#Day19 直方图比较 
v = cv.compareHist(hist1,hist2,methods) #cv.HISTCMP_BHATTACHARYYA 巴氏距离;cv.HISTCMP_CORREL相关性;cv.HISTCMP_CHISQR卡方 cv.HISTCMP_HELLINGER交叉
'''
#Day20 图像直方图反向投影：通过构建指定模板图像的二维直方图空间与目标的二维直方图空间，进行直方图数据归一化之后， 进行比率操作，对所有得到非零数值，生成查找表对原图像进行像素映射之后，再进行图像模糊输出的结果。

cv.waitKey(0)
cv.destroyAllWindows()


