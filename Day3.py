import cv2 as cv
import numpy as np

src = cv.imread("test.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input",src)
'''
#Day21 图像卷积操作
def custom_blur(src): #自己编的卷积操作
    h, w, ch = src.shape
    print("h , w, ch", h, w, ch)
    result = np.copy(src)
    for row in range(1, h-1, 1):
        for col in range(1, w-1, 1):
            v1 = np.int32(src[row-1, col-1])
            v2 = np.int32(src[row-1, col])
            v3 = np.int32(src[row-1, col+1])
            v4 = np.int32(src[row, col-1])
            v5 = np.int32(src[row, col])
            v6 = np.int32(src[row, col+1])
            v7 = np.int32(src[row+1, col-1])
            v8 = np.int32(src[row+1, col])
            v9 = np.int32(src[row+1, col+1])

            b = v1[0] + v2[0] + v3[0] + v4[0] + v5[0] + v6[0] + v7[0] + v8[0] + v9[0];
            g = v1[1] + v2[1] + v3[1] + v4[1] + v5[1] + v6[1] + v7[1] + v8[1] + v9[1];
            r = v1[2] + v2[2] + v3[2] + v4[2] + v5[2] + v6[2] + v7[2] + v8[2] + v9[2];
            result[row, col] = [b//9, g//9, r//9]
    cv.imshow("result", result)

dst = cv.blur(src, (15, 15)) #卷积函数（均值滤波，15*15卷积核）
# custom_blur(src)
cv.imshow("blur", dst)

#Day22/23 均值滤波和高斯滤波,中值滤波
dst1=cv.GaussianBlur(src,(15,15),sigmaX=0) #size(0,0)当Size(0, 0)就会从sigmax开始计算生成高斯卷积核系数，当时size不为零是优先从size开始计算高斯卷积核系数
cv.imshow("",dst1)
dst2 = cv.medianBlur(src, 5) #中值滤波（src,ksize必须是奇数，而且必须大于1），对椒盐噪声又好的去噪效果

#Day24 噪声:椒盐噪声；高斯噪声；均匀分布噪声

#Day25-27,29 图像卷积功能一：图像模糊/去噪：图像去噪声在OCR、机器人视觉与机器视觉领域应用，OpenCV中常见的图像去噪声的方法有
#- 均值去噪声 cv.blur()
#- 高斯模糊去噪声 cv.GaussianBlur()
#- 非局部均值去噪声
#- 边缘保留滤波算法EPF：高斯双边滤波，Meanshift均值迁移，局部均方差
#- 形态学去噪声
cv.fastNlMeansDenoisingColored(src,None,15,15,10,30) #非局部均值去噪 *
cv.bilateralFilter(src,0,100,10) #高斯双边滤波 *
cv.pyrMeanShiftFiltering(src,15,30,termcrit=(cv.TERM_CRITERIA_MAX_ITER+cv.TERM_CRITERIA_EPS,5,1)) # Meanshift均值迁移
cv.edgePreservingFilter(src,sigma_s=100,sigma_r=0.4,flags=cv.RECURS_FILTER)#图像边缘保留滤波算法，实现了数据降维，比前面两种计算速度快。
'''
#Day28 图像积分图算法：图像积分图在图像特征提取HAAR/SURF、二值图像分析、图像相似相关性NCC计算、图像卷积快速计算等方面均有应用，是图像处理中的经典算法之一。

#Day30 opencv自定义滤波器:自定义卷积核常见的主要是均值、锐化、梯度等算子
blur_op = np.ones([5, 5], dtype=np.float32)/25. #均值模糊
shape_op = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], np.float32) #锐化
grad_op = np.array([[1, 0],[0, -1]], dtype=np.float32) #梯度

dst1 = cv.filter2D(src, -1, blur_op) #默认-1，表示输入与输出图像类型一致，但是当涉及浮点数计算时候，需要设置为CV_32F
dst2 = cv.filter2D(src, -1, shape_op)
dst3 = cv.filter2D(src, cv.CV_32F, grad_op)
dst3 = cv.convertScaleAbs(dst3) #因为CV_32E，滤波完成之后需要使用convertScaleAbs函数将结果转换为字节类型。

cv.imshow("blur=5x5", dst1);
cv.imshow("shape=3x3", dst2);
cv.imshow("gradient=2x2", dst3);

cv.waitKey(0)
cv.destroyAllWindows()

