import cv2 as cv
import numpy as np

src = cv.imread("3.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
'''
#Day31-34，36 卷积功能二：梯度/边缘提取:一阶微分算子robert，prewitt;Sobel,scharr，二阶微分（拉普拉斯
#Sobel
h, w = src.shape[:2]
x_grad = cv.Sobel(src, cv.CV_32F, 1, 0) #Sobel(src,图像深度，X方向一阶导数，Y方向)
y_grad = cv.Sobel(src, cv.CV_32F, 0, 1)
x_grad = cv.convertScaleAbs(x_grad) #放缩为0-255 8位的图像，之后再相加，可以减少噪声
y_grad = cv.convertScaleAbs(y_grad)
#cv.imshow("x_grad", x_grad)
#cv.imshow("y_grad", y_grad)
dst = cv.add(x_grad, y_grad, dtype=cv.CV_16S) #X梯度与Y梯度相加，得到梯度图，也就是边缘图
dst = cv.convertScaleAbs(dst)
cv.imshow("Sobel", dst)
#自定义滤波函数实现robert,prewitt
robert_x = np.array([[1,0],[0,-1]],dtype=np.float32) #卷积模板
robert_y = np.array([[0,-1],[1,0]],dtype=np.float32)
#prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
#prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
robert_x_grad = cv.filter2D(src,cv.CV_16S,robert_x)
robert_y_grad = cv.filter2D(src,cv.CV_16S,robert_y)
robert_x_grad = cv.convertScaleAbs(robert_x_grad)
robert_y_grad = cv.convertScaleAbs(robert_y_grad)
result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h,0:w,:] = robert_x_grad
result[0:h,w:2*w,:] = robert_y_grad
cv.imshow("robert", result)
#拉普拉斯算子（二阶导数算子，二阶导数是求X、Y方向的二阶偏导数），突出图片纹理
src = cv.GaussianBlur(src,(0,0),1) #先高斯模糊,sigmaX=1
lap = cv.Laplacian(src,cv.CV_32F,ksize=3,delta=127) #ksize等于1是四邻域算子，大于1改用八邻域算子,delate对输出图像加上常量值
lap = cv.convertScaleAbs(lap)
cv.imshow("laplacian", lap)
#Canny算子:高斯模糊+梯度提取X/Y平方和+角度计算和非最大信号抑制+高低阈值链接，获取完整边缘=输出边缘
edge = cv.Canny(src,100,300) #高低阈值，高低阈值之比在2:1～3:1之间
cv.imshow("canny",edge)
edge_src = cv.bitwise_and(src,src,mask=edge) #得到彩色边缘
cv.imshow("",edge_src)

#Day34,35 卷积功能三：锐化/增强：
#基于拉普拉斯滤波的锐化算子
# sharpen_op = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32) #8领域
sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32) #4领域
sharpen_image = cv.filter2D(src, cv.CV_32F, sharpen_op)
sharpen_image = cv.convertScaleAbs(sharpen_image)
cv.imshow("sharpen_image", sharpen_image)
#显示并添加文字标注
h, w = src.shape[:2]
sharp = np.zeros([h, w*2, 3], dtype=src.dtype)
sharp[0:h,0:w,:] = src
sharp[0:h,w:2*w,:] = sharpen_image
cv.putText(sharp, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2) #添加文本
cv.putText(sharp, "sharpen image", (w+10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.imshow("sharpen_image", sharp)
cv.imwrite("D:/result.png", sharp)
#USM(Unsharpen Mask)锐化：可以除去干扰细节和噪声，提升锐化效果
#（源图像– w*高斯模糊）/（1-w）；其中w表示权重（0.1～0.9），默认为0.6
blur_img = cv.GaussianBlur(src,(0,0),5)
usm = cv.addWeighted(src,1.5,blur_img,-0.5,0)#基于权重的加减，（src*1.5+blur*-0.5），0表示否伽马校正
cv.imshow("mask image",usm)

#Day37，38 图像金字塔 ：可用于特征提取，空间尺度不变性特征
# 高斯金字塔 :reduce下采样 和 expend上采样，要逐层放缩，一次操作放缩1/4。先高斯模糊再上下采样
def pyrDown(image, level=3): #reduce
    temp = image.copy()
    # cv.imshow("input", image)
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp) #下采样
        pyramid_images.append(dst)
        # cv.imshow("pyr_down_" + str(i), dst)
        temp = dst.copy()
    return pyramid_images  #得到三层金字塔图集
def pyrUp(pyramid_images): #expand
    level = len(pyramid_images)
    print("level = ",level)
    for i in range(level-1, -1, -1):
        expand = cv.pyrUp(pyramid_images[i])  #上采样expand
        cv.imshow("pyr_up_"+str(i), expand)
pyrUp(pyrDown(src))
#拉普拉斯金字塔：相应大小层的reduce图减去expand图的图像，得到两次高斯模糊输出的不同。
#应用：知道高斯金字塔的下采样图和拉普拉斯金字塔，可以构造出原图
def laplaian_demo(pyramid_images):
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0: #当 i=0 ，没有i-1层，所以单独讨论
            h, w = src.shape[:2]
            expand = cv.pyrUp(pyramid_images[i], dstsize=(w, h)) #dstsize限定上采样的大小，便于之后的相减操作
            lpls = cv.subtract(src, expand) + 127 #加127是为了方便显示，实验过程的中间结果不能这样处理
            cv.imshow("lpls_" + str(i), lpls)
        else:
            h, w = pyramid_images[i-1].shape[:2]
            expand = cv.pyrUp(pyramid_images[i], dstsize=(w, h))
            lpls = cv.subtract(pyramid_images[i-1], expand) + 127
            cv.imshow("lpls_"+str(i), lpls)
laplaian_demo(pyrDown(src))

#Day39 模板匹配：入门级模式识别方法，有着严格的理论条件。
# cv.matchTemplate(	image, templ, method[, result[, mask]] )
# 方法有cv.TM_SQDIFF = 0 平方不同
# TM_SQDIFF_NORMED = 1 平方不同的归一化
# TM_CCORR = 2 相关性，值越大相关性越强，表示匹配程度越高。
# TM_CCORR_NORMED = 3 归一化版本值在0～1之间，1表示高度匹配，0表示完全不匹配
# # TM_CCOEFF = 4 相关因子，值越大相关性越强，表示匹配程度越高
# TM_CCOEFF_NORMED = 5 归一化版本值在0～1之间，1表示高度匹配，0表示完全不匹配
tpl = cv.imread("39_pl.png")
result = cv.matchTemplate(src,tpl,cv.TM_CCORR_NORMED) #得到的是匹配的数值
#cv.imshow("",np.uint8(result*255)) 显示不出来??

#Day40 图像分析之二值图像：二值图像分析包括轮廓分析、对象测量、轮廓匹配与识别、形态学处理与分割、各种形状检测与拟合、投影与逻辑操作、轮廓特征提取与编码等
# 转换为灰度图像
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
h, w = gray.shape
T = cv.mean(gray)[0] #像素的均值作为阈值
print("current threshold value : ", T)
binary = np.zeros((h, w), dtype=np.uint8)
for row in range(h):
    for col in range(w):
        pv = gray[row, col]
        if pv > T:
            binary[row, col] = 255
        else:
            binary[row, col] = 0
cv.imshow("binary", binary)

#Day41-44 阈值选择操作算法
# 基本阈值操作
# THRESH_BINARY = 0 二值分割
# THRESH_BINARY_INV = 1 反向二值分割
# THRESH_TRUNC = 2 截断
# THRESH_TOZERO = 3 取零
# THRESH_TOZERO_INV = 4 反向取零
T=127
for i in range(5):
    ret, binary = cv.threshold(gray, T, 255, i) #i代表了阈值操作方法
    cv.imshow("binary_" + str(i), binary)
# OTSU 通过计算类间最大方差来确定分割阈值的阈值选择算法,对有两个波峰之间有一个波谷的直方图效果好
ret1,ostu = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
print("ret1",ret1)
cv.imshow("ostu",ostu)
#TRIANGLE:适用于直方图只有一个波峰
ret2,triangle = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
print("ret2",ret2)
cv.imshow("triangle",triangle)
#自适应阈值算法：主要是基于均值实现，根据计算均值的方法不同分为盒子模糊均值与高斯模糊均值.。提取到的更像是边缘，示例上是文字提取。
# InputArray src,
# double maxValue,
# int adaptiveMethod, ADAPTIVE_THRESH_GAUSSIAN_C = 1;ADAPTIVE_THRESH_MEAN_C = 0
# int thresholdType, THRESH_BINARY:原图-均值图>-c?255:0  HRESH_BINARY_INV与前面的想反
# int blockSize, 取值必须是奇数
# double C 取值在10左右
adapt = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,10)
cv.imshow("adapt",adapt)
'''
#Day45 图像二值化与去噪 ：二值化之前可以先去噪（之前的去噪方法）

cv.waitKey(0)
cv.destroyAllWindows()
