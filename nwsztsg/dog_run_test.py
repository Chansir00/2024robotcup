# -*- coding: utf-8 -*-
import time
import struct
import cv2 as cv
import numpy as np
import threading
import os
import argparse

from utils.Controller import Controller

client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)

global frame

if __name__ == '__main__':
    # creat a controller
    controller = Controller(server_address)
    stop_heartbeat = False
    
    cap = cv.VideoCapture("/dev/video4", cv.CAP_V4L2)
    
    cap.set(3, 640)
    cap.set(4, 480)
    # 图像计数 从1开始
    img_count = 1
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
    
    step_change_flag = 0
    runmode_change_flag = 0

    while(1):
        ret, frame = cap.read()
        if ret:
            # show a frame
            cv.imshow("capture", frame)
            # 等待按键事件发生 等待1ms
            key = cv.waitKey(1)
            if key == ord('q'):
                # 如果按键为q 代表quit 退出程序
                print("程序正常退出..")
                break
            elif key == ord('e'):  # 
                ## 如果s键按下，则进行图片保存
                if step_change_flag == 0:
                    pack = struct.pack('<3i', 0x21010401, 0, 0)
                    controller.send(pack)
                    # time.sleep(5)
                    print('step change')
                    step_change_flag = 1
                else:
                    pack = struct.pack('<3i', 0x21010304, 0, 0)
                    controller.send(pack)
                    # time.sleep(5)
                    print('step change back')
                    step_change_flag = 0
                    
                
            elif key == ord('w'):  #     
                pack = struct.pack('<3i', 0x21010130, 17000, 0)
                controller.send(pack)
                # time.sleep(3)
                print('go')
                
            elif key == ord('a'):  # 
                ## 如果s键按下，则进行图片保存
                pack = struct.pack('<3i', 0x21010135, -13000, 0)
                controller.send(pack)
                # time.sleep(5)
                print('left')
                
            elif key == ord('d'):  # 
                ## 如果s键按下，则进行图片保存
                pack = struct.pack('<3i', 0x21010135, 13000, 0)
                controller.send(pack)
                # time.sleep(5)
                print('right')
                
            elif key == ord('x'):
                pack = struct.pack('<3i', 0x21010130, -10000, 0)
                controller.send(pack)
                # time.sleep(3)
                print('back')
                
            elif key == ord('s'):
                pack = struct.pack('<3i', 0x21010130, 0, 0)
                controller.send(pack)
                pack = struct.pack('<3i', 0x21010135, 0, 0)
                controller.send(pack)
                # time.sleep(3)
                print('stop')
                
            elif key == ord('r'):
                if runmode_change_flag == 0:
                    # stop the dog
                    pack = struct.pack('<3i', 0x21010130, 0, 0)
                    controller.send(pack)
                    pack = struct.pack('<3i', 0x21010135, 0, 0)
                    controller.send(pack)
                    print('stay_mode')
                    
                    # noding
                    pack = struct.pack('<3i', 0x21010D05, 0, 0)  # staymode
                    controller.send(pack)
                    pack = struct.pack('<3i', 0x21010130, -32000, 0)
                    controller.send(pack)
                    time.sleep(1)
                    pack = struct.pack('<3i', 0x21010130, 32000, 0)
                    controller.send(pack)
                    time.sleep(1)
                    pack = struct.pack('<3i', 0x21010130, 0, 0)
                    controller.send(pack)
                    time.sleep(1)

                    pack = struct.pack('<3i', 0x21010135, -32000, 0)
                    controller.send(pack)
                    time.sleep(1)
                    pack = struct.pack('<3i', 0x21010135, 32000, 0)
                    controller.send(pack)
                    time.sleep(1)
                    pack = struct.pack('<3i', 0x21010135, 0, 0)
                    controller.send(pack)
                    time.sleep(1)
                    # pack = struct.pack('<3i', 0x21010135, 32000, 0)
                    # controller.send(pack)
                    # pack = struct.pack('<3i', 0x21010135, -32000, 0)
                    # controller.send(pack)
                    # pack = struct.pack('<3i', 0x21010135, 32000, 0)
                    # controller.send(pack)
                    # pack = struct.pack('<3i', 0x21010135, -32000, 0)
                    # controller.send(pack)
                    # pack = struct.pack('<3i', 0x21010135, 0, 0)
                    # controller.send(pack)
                    print('noding')
                    runmode_change_flag = 1
                    
                else:
                    pack = struct.pack('<3i', 0x21010D06, 0, 0)  # staymode
                    controller.send(pack)
                    pack = struct.pack('<3i', 0x21010130, 0, 0)
                    controller.send(pack)
                    pack = struct.pack('<3i', 0x21010135, 0, 0)
                    controller.send(pack)
                    time.sleep(3)
                    print('run_mode')
                    runmode_change_flag = 0
        else:
            print("图像数据获取失败！！")
            break

    controller.drive_dog("squat")
    cap.release()
    cv.destroyAllWindows()
    stop_heartbeat = True

