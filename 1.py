import cv2 as cv
import numpy as np

src = cv.imread("test.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 蓝色通道为零
mv = cv.split(src)
mv[0][:, :] = 0
dst1 = cv.merge(mv)
cv.imshow("output1", dst1)

# 绿色通道为零
mv = cv.split(src)
mv[1][:, :] = 0
dst2 = cv.merge(mv)
cv.imshow("output2", dst2)

# 红色通道为零
mv = cv.split(src)
mv[2][:, :] = 0
dst3 = cv.merge(mv)
cv.imshow("output3", dst3)

dst = np.zeros(src.shape, dtype=np.uint8)
print(src.shape)
print(dst.shape)
cv.mixChannels([src], [dst], fromTo=[2, 0, 1, 1, 0, 2])
cv.imshow("output4", dst)

cv.waitKey(0)
cv.destroyAllWindows()