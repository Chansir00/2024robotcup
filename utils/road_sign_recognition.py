import cv2 as cv
import numpy as np
import threading
import struct
import time

from utils.RoadSignRecognition import RoadSignRecognition
from utils.Controller import Controller

Develop_Mode = True # True means use computer camera. False means use dog camera
client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)

if __name__ == '__main__':
    sign_detector = RoadSignRecognition()
    controller = Controller(server_address)

    if Develop_Mode:
        cap = cv.VideoCapture(0)
        #frame = cv.imread('right.jpg')
    else:
        cap = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        bbox = sign_detector.detect(frame)
        frame = sign_detector.visualize(frame)
        sign = sign_detector.classify(frame)
        sign_detector.update_results(sign)

        # 每次检测后获取最常见的信号
        common_sign = sign_detector.most_common_sign()
        if common_sign is not None:
            if common_sign  == '1':
                print('turn left')
            elif common_sign == '0':
                print('turn right')
        cv.imshow("Road Sign Recognition", frame)
        k = cv.waitKey(1)
        if k == 113 or k == 81:  # q or Q to quit
            print("Demo is quitting......")
            if not Develop_Mode:
                controller.drive_dog("squat")
            cap.release()
            cv.destroyWindow("Road Sign Recognition")
            stop_heartbeat = True
            break
    # if frame is not None:
    #     bbox = sign_detector.detect(frame)
    #     frame = sign_detector.visualize(frame)
    #     sign = sign_detector.classify(frame)
    #     if sign is not None:
    #         print(sign)
    #     cv.imshow("Road Sign Recognition", frame)
    #     cv.waitKey(0)  # 等待按键后关闭
    #     cv.destroyAllWindows()
    # else:
    #     print("Image not found or unable to read.")