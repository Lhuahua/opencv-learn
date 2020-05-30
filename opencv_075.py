import cv2 as cv
#图像去水印/修复
if __name__ == '__main__':
    src = cv.imread("D:/images/master2.jpg")
    cv.imshow("watermark image", src)
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (100, 43, 46), (124, 255, 255))
    cv.imshow("mask", mask)
    cv.imwrite("D:/mask.png", mask)
    se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    cv.dilate(mask, se, mask) #膨胀，变白
    result = cv.inpaint(src, mask, 3, cv.INPAINT_TELEA) #（输入图，输入mask（修复非零区域）,paintRadius，修复方法）
    cv.imshow("result", result)
    cv.imwrite("D:/result.png", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
