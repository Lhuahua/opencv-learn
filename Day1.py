import cv2 as cv
import numpy as np

#学习笔记From  opencv研习社-贾志刚
#图片文件在：https://github.com/gloomyfish1998/opencv_tutorial/tree/master/data/images

src = cv.imread("test.jpg") #读
'''
Day 1-2
cv.namedWindow("Input",cv.WINDOW_AUTOSIZE)
cv.imshow("input",src) #显示
gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY) #灰度。cvtColor：色彩空间转换函数
cv.imshow("gray",gray)
cv.imwrite("gray.jpg",gray)


Day3 对象的创建和赋值
m1 = np.copy(src)
m2 = src
src[100:200,200:300,:] = 255
cv.imshow("m2",m2)
m3 = np.zeros(src.shape,src.dtype)
m3[:,:,0] = 255 #图像的第一通道（Blue）为255； 0（B） 1（G） 2（R）
cv.imshow("m3",m3)

Day4-5 像素的读写,运算-加减乘除，交并补（加减直接数组运算，乘除要像素运算；此外要检查数据越界0-255）
h,w,ch = src.shape
print("h,w,ch",h,w,ch)
for row in range(h):
    for col in range(w):
        b,g,r=src[row,col]
        b = 255 - b
        g = 255 - g
        r = 255 - r
        src[row,col] = [b,g,r]
cv.imshow("output",src)

#Day6  Look Up Table(LUT)查找表，用于颜色匹配和色彩增强（低对比度图映射成高对比度图），转换成二值图像
dst = cv.applyColorMap(src,cv.COLORMAP_HSV) #常用于灰度图的涂色
cv.imshow("colurmap",dst)

#DAY7 像素的逻辑操作（位操作）：与 或 异或 非
src1 = np.zeros(shape=[400,400,3],dtype=np.uint8)
src1[100:200,100:200,1] = 255 #颜色为绿色矩形
src1[100:200,100:200,2] = 255 #颜色成黄色
src2 = np.zeros(shape=[400,400,3],dtype=np.uint8)
src2[150:250,150:250,2] = 255
cv.imshow("src1",src1)
cv.imshow("src2",src2)
dst1 = cv.bitwise_and(src1,src2) #cv.bitwise_not(src);cv.bitwise_xor();
cv.imshow("dst1",dst1)

#Day8 通道分离与合并
mv = cv.split(src) #分离 BGR三通道
mv[0][:] = 0 #B通道都为0
dst1 = cv.merge(mv) #合成的图像 去掉了蓝色通道，变成主色调为黄色的图（红+绿=黄） ;
# mixChannels()：把输入的矩阵（或矩阵数组）的某些通道拆分复制给对应的输出矩阵（或矩阵数组）的某些通道中，其中的对应关系就由fromTo参数制定.
cv.imshow("dst1",dst1)

#Day9 色彩空间：RGB,HSV直方图,YUV,YCrCb做皮肤检测
hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
mask = cv.inRange(hsv,(35,43,46),(99,255,255)) #提取指定色彩范围区域（输入图像，通道颜色低值，高值）;输出：符合范围的点为白色255，不符合的为黑色0
cv.imshow("",mask)
#dst = cv.bitwise_and(src,src,mask=mask) #src与src先与在于mask与

#Day10 像素值统计：

gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
min, max, minLoc, maxLoc = cv.minMaxLoc(gray) # cv.minMaxLoc(src) 返回 min/max/minLoc/maxLoc
print("min: %.2f, max: %.2f"% (min, max))
print("min loc: ", minLoc)
print("max loc: ", maxLoc)
means, stddev = cv.meanStdDev(gray) # cv.meanStdDev(src) 返回 means，stddev方差
print("mean: %.2f, stddev: %.2f"% (means, stddev))
gray[np.where(gray < means)] = 0
gray[np.where(gray > means)] = 255
cv.imshow("binary",gray) #均值滤波
'''
cv.waitKey(0) #waitKey(0)表示阻塞等待用户键盘输入，用户按键盘任意键就会停止阻塞，继续执行直到程序正常退出！
cv.destroyAllWindows()#