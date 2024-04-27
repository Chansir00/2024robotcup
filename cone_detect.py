import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


def update(x):
    global gs, erode, Hmin, Smin, Vmin, Hmax, Smax, Vmax, img, Hmin2, Hmax2, img0, size_min,color
    ret, img0 = cap.read()
    gs = 0
    erode = 0
    Hmin = 0
    Hmax = 125
    Hmin2 = 0
    Hmax2 = 0
    Smin = 50
    Smax = 255
    Vmin = 141
    Vmax = 240
    size_min = 9000
    size_max = 110000

    # 滤波二值化
    gs_frame = cv2.GaussianBlur(img0, (gs, gs), 1)
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)
    erode_hsv = cv2.erode(hsv, None, iterations=erode)
    inRange_hsv = cv2.inRange(erode_hsv, np.array([Hmin, Smin, Vmin]), np.array([Hmax, Smax, Vmax]))
    inRange_hsv2 = cv2.inRange(erode_hsv, np.array([Hmin2, Smin, Vmin]), np.array([Hmax2, Smax, Vmax]))
    img = inRange_hsv + inRange_hsv2
    # 外接计算
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    target_list = []
    pos = []
    if size_min < 1:
        size_min = 1
    for c in cnts:
        if cv2.contourArea(c) < size_min:
            continue
        else:
            target_list.append(c)
    for cnt in target_list:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # pos.append([int(x + w / 2), y + h / 2])
        # 计算矩形中心坐标
        center_x = x + w // 2
        center_y = y + h // 2

        # 将图像转换为 HSV 色彩空间
        hsv_img = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)

        # 获取中心点的 HSV 值
        center_hsv = hsv_img[center_y, center_x]

        # print("HSV value at the center of the rectangle:")
        # print(center_hsv)
        if center_hsv[0]>50:
            color = 1
        elif center_hsv[0]>15:
            color = 2
        else:
            color = 0
        return color


mode = 'camera'

if mode == 'camera':
    cap = cv2.VideoCapture(0)

elif mode == 'picture':
    img0 = cv2.imread('test.jpg')



if __name__ == '__main__':
    while (True):
        update(1)
        # cv2.imshow('image', img)
        # cv2.imshow('image1', img)
        cv2.imshow('image0', img0)
        time.sleep(0.1)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


