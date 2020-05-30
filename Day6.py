import cv2 as cv
import numpy as np

src = cv.imread("test.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# Day62 图像形态学 - 膨胀与腐蚀，可用于二值图，灰度图，色彩图
se = np.ones((3, 3), dtype=np.uint8) # 使用3x3结构元素进行膨胀与腐蚀操作
dilate = cv.dilate(src, se, None, (-1, -1), 1) #膨胀可以看成是最大值滤波，即用最大值替换中心像素点；
erode = cv.erode(src, se, None, (-1, -1), 1) # 腐蚀可以看出是最小值滤波，即用最小值替换中心像素点
cv.imshow("dilate", dilate)
cv.imshow("erode", erode)

#结构元素的选择 cv.getStructuringElement
# int shape #cv.MORPH_** 指结构元素的类型，常见的有矩形、圆形、十字交叉
# Size ksize,
# Point anchor = Point(-1,-1)
gray = cv.cvtColor(src,cv.COLOR_RGB2GRAY)
ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU) #二值化
se = cv.getStructuringElement(cv.MORPH_RECT,(3,3),(-1,-1)) #3*3大小的矩形结构元素
dilate = cv.dilate(src, se, None, (-1, -1), 1)
erode = cv.erode(src, se, None, (-1, -1), 1)
#开操作： 腐蚀 + 膨胀，可以删除二值图像中小的干扰块，降低图像二值化之后噪点过多的问题。morphologyEx（）MORPH_OPEN
#闭操作：膨胀 + 腐蚀，可以填充二值图像中孔洞区域，形成完整的闭合区域连通组件 morphologyEx（）MORPH_CLOSE
#顶帽操作：原图 – 开操作，取图像中微小部分特别有用 MORPH_TOPHAT
#黑帽操作 = 闭操作 – 输入图像 MORPH_BLACKHAT
#基本梯度操作：图像膨胀与腐蚀操作之间的差值，实现图像轮廓或者边缘提取 MORPH_GRADIEN
#内梯度是输入图像与腐蚀之间的差值 MORPH_DILATE
#外梯度是膨胀与输入图像之间的差值 MORPH_ERODE
#击中击不中操作:MORPH_HITMISS

# src 输入图像
# op 形态学操作 op指定为MORPH_OPEN 表示开操作，MORPH_CLOSE表示闭操作，MORPH_TOPHAT表示顶帽操作，MORPH_BLACKHAT表示使用黑帽操作，MORPH_GRADIEN即表示使用基本梯度操作
# kernel 结构元素
# anchor 中心位置锚定
# iterations 循环次数
# borderType 边缘填充类型
morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,se)
morph_close = cv.morphologyEx(binary,cv.MORPH_CLOSE,se)
#Day66 开操作提取二值图像中水平与垂直线，闭操作实现不同层次的轮廓填充
#Day70 使用基本梯度实现轮廓分析:基于形态学梯度实现图像二值化，进行文本结构分析是OCR识别中常用的处理手段之一，这种好处比简单的二值化对图像有更好的分割效果，
#Day72,73 项目 缺陷检测
#Day74 提取最大轮廓与编码关键点
#Day75 图像去水印 inpaint
#Day76 图像透视变换应用
cv.waitKey(0)
cv.destroyAllWindows()
