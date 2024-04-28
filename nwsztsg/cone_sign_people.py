# -*- coding: utf-8 -*-
import time
import struct
import cv2
import numpy as np
import threading
import os
import argparse
import pyttsx3

import onnxruntime

from utils.Controller import Controller

client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)

CLASSES = ['people']  # coco80类别
CLASSES1 = ['塌方','爆炸','冒顶','火灾','水灾']
CLASSES2 = ['红色','蓝色','黄色']


global frame,Develop_Mode,sign_mode ,people_mode,cone_mode

Develop_Mode = True
sign_mode = True
people_mode = True
cone_mode = True

class YOLOV5():
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath, providers=["CUDAExecutionProvider"])
        # self.onnx_session=onnxruntime.InferenceSession(onnxpath, providers=["TensorrtExecutionProvider"])
        # self.onnx_session=onnxruntime.InferenceSession(onnxpath, providers=["CPUExecutionProvider"])
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    # -------------------------------------------------------
    #   获取输入输出的名字
    # -------------------------------------------------------
    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    # -------------------------------------------------------
    #   输入图像
    # -------------------------------------------------------
    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        return input_feed

    # -------------------------------------------------------
    #   1.cv2读取图像并resize
    #	2.图像转BGR2RGB和HWC2CHW
    #	3.图像归一化
    #	4.图像增加维度
    #	5.onnx_session 推理
    # -------------------------------------------------------
    # def inference(self, img_path):
    #     img = cv2.imread(img_path)
    #     or_img = cv2.resize(img, (640, 640))
    #     img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
    #     img = img.astype(dtype=np.float32)
    #     img /= 255.0
    #     img = np.expand_dims(img, axis=0)
    #     input_feed = self.get_input_feed(img)
    #     pred = self.onnx_session.run(None, input_feed)[0]
    #     return pred, or_img
    def inference(self, frame):
        # img = cv2.imread(img_path)
        or_img = cv2.resize(frame, (640, 640))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, or_img


# dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
# thresh: 阈值
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # -------------------------------------------------------
    #   计算框的面积
    #	置信度从大到小排序
    # -------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        # -------------------------------------------------------
        #   计算相交面积
        #	1.相交
        #	2.不相交
        # -------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # -------------------------------------------------------
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #	IOU小于thresh的框保留下来
        # -------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def filter_box(org_box, conf_thres, iou_thres):  # 过滤掉无用的框
    # -------------------------------------------------------
    #   删除为1的维度
    #	删除置信度小于conf_thres的BOX
    # -------------------------------------------------------
    org_box = np.squeeze(org_box)
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]
    # -------------------------------------------------------
    #	通过argmax获取置信度最大的类别
    # -------------------------------------------------------
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        try:
            cls.append(int(np.argmax(cls_cinf[i])))
        except:
            continue
    all_cls = list(set(cls))
    # -------------------------------------------------------
    #   分别对每个类别进行过滤
    #	1.将第6列元素替换为类别下标
    #	2.xywh2xyxy 坐标转换
    #	3.经过非极大抑制后输出的BOX下标
    #	4.利用下标取出非极大抑制后的BOX
    # -------------------------------------------------------
    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])
        curr_cls_box = np.array(curr_cls_box)
        # curr_cls_box_old = np.copy(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = nms(curr_cls_box, iou_thres)
        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output


def draw(image, box_data):
    # -------------------------------------------------------
    #	取整，方便画框
    # -------------------------------------------------------
    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)

    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES1[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES1[cl], score),
                    (top, left),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def yuy(people_count=0,cone=0,sign=0):
    global people_mode
    engine = pyttsx3.init()
    engine.setProperty('voice', 'zh') #开启支持中文
    # engine.say('检测到'+str(people_count)+'人')
    # engine.say("{}区域发现{},受困人数{}人".format(CLASSES2[cone],CLASSES1[sign],people_count))
    print("{}区域发现{},受困人数{}人".format(CLASSES2[cone],CLASSES1[sign],people_count))
    engine.runAndWait()
    engine.stop()
    people_mode = False

#识别标识牌
result_list = []
def sign_detect(input_list):
    global sign_mode
    max_list_num = 10
    max_num = 0
    max_order = 0
    input_list = list(input_list[0])
    for i in range(5):
        if input_list[i] <0:
            continue
        if i != 3:
            if input_list[i]>max_num:
                max_order = i;
                max_num = input_list[max_order]
        else:
            if input_list[i] - 0.5>max_num:
                max_order = i;
                max_num = input_list[max_order]
    result_list.append(max_order)
    if len(result_list)>max_list_num:
        result_list.pop(0)
        result = max(set(result_list),key=result_list.count)
        sign_mode = False
        return result

#识别锥桶
result_list2 = []
def update(x):
    global gs, erode, Hmin, Smin, Vmin, Hmax, Smax, Vmax, img, Hmin2, Hmax2, img0, size_min,cone_mode
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
        max_list_num = 10
        if center_hsv[0]>50:
            result_list2.append(1)    #蓝色
        elif center_hsv[0]>15:
            result_list2.append(2)     #黄色
        else:
            result_list2.append(0)   #红色
        if len(result_list2)>max_list_num:
            result_list2.pop(0)
            result = max(set(result_list2),key=result_list2.count)
            cone_mode = False
            return result








if __name__ == '__main__': 
    last_people_cnt = 0;   
    # 创建控制器
    controller = Controller(server_address)
    stop_heartbeat = False

    # cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)


    # cap = cv.VideoCapture(0)
    # cap.set(3, 640)
    # cap.set(4, 480)
    # 图像计数 从1开始

    # start to exchange heartbeat pack
    if Develop_Mode:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        stop_heartbeat = False
        # start to exchange heartbeat pack
        def heart_exchange(con):
            pack = struct.pack('<3i', 0x21040001, 0, 0)
            while True:
                if stop_heartbeat:
                    return
                con.send(pack)
                time.sleep(0.25)  # 4Hz

        heart_exchange_thread = threading.Thread(target=heart_exchange, args=(controller,))
        heart_exchange_thread.start()

        # stand up
        print("Wait 10 seconds and stand up......")
        pack = struct.pack('<3i', 0x21010202, 0, 0)
        controller.send(pack)
        time.sleep(5)
        controller.send(pack)
        time.sleep(5)
        controller.send(pack)
        print("Dog should stand up, otherwise press 'ctrl + c' and re-run the demo")
        time.sleep(5)

        pack = struct.pack('<3i', 0x21010D05, 0, 0)
        controller.send(pack)
        # time.sleep(5)
        print('stop ges')

        pack = struct.pack('<3i', 0x21010130, -32000, 0)
        controller.send(pack)
        time.sleep(3)
        print('head up')


    onnx_path1 = r"X:\robot\2024\nwsztsg\sign_best4.onnx"    #危险标志
    onnx_path2 = r"X:\robot\2024\nwsztsg\sign_best4.onnx"    #人数
    print('model is loading......')
    model1 = YOLOV5(onnx_path1)     #加载危险标志模型
    model2 = YOLOV5(onnx_path2)        #加载人物识别模型
    print('model loads successfully!')

    #python export.py --weights runs/train/exp7/best.pt --include onnx --opset 12 --dynamic

    while (1):
        # if ret:
        #     # show a frame
        #     # start_time = time.time()
        ret, frame = cap.read()
            #锥桶识别
        if cone_mode:
            while True:
                ret, frame = cap.read()
                img0 = frame
                cone = update(1)
                if cone_mode == False:
                    break
        #危险标志识别
        if sign_mode:
            while True:
                ret, frame = cap.read()
                output, or_img = model2.inference(frame)
                sign = sign_detect(output)
                if sign_mode == False:
                    break
        # 人数识别
        if people_mode:
            while True:
                ret, frame = cap.read()
                output, or_img = model1.inference(frame)  # 模型推理
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print("Elapsed time:", elapsed_time, "seconds")
                outbox = filter_box(output, 0.5, 0.5)
                people_count = len(outbox)
                if people_count > 0:
                    draw(or_img, outbox)
                    if people_count != last_people_cnt:
                        last_people_cnt = len(outbox)
                yuy(people_count,cone,sign)
                if people_mode == False:
                    break


        cv2.imshow("infer", frame)
        # 等待按键事件发生 等待1ms
        key = cv2.waitKey(1)
        if key == ord('q'):
            # 如果按键为q 代表quit 退出程序
            print("程序正常退出..")
            break
        elif key == ord('c'):
            print("重新检测")
            sign_mode = True
            result_list.clear()
            people_mode = True
            result_list2.clear()
            cone_mode = True
        # else:
        #     print("图像数据获取失败！！")
        #     break

    controller.drive_dog("squat")
    cap.release()
    cv2.destroyAllWindows()
    stop_heartbeat = True
