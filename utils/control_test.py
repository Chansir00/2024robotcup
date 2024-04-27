# 机器狗起立前校准
import time
import struct
import threading
import os
# import psutil
import cv2 as cv
import numpy as np

from danger_alarm import DangerSignRecognition
from heartbeat import Heartbeat

from utils.Controller import Controller
from opencv_zoo.models.text_detection_db.db import DB
from opencv_zoo.models.text_recognition_crnn.crnn import CRNN
from utils.DangerSignRecognition import DangerSignRecognition

os.system(f'sudo clear')  # 引导用户给予root权限，避免忘记sudo运行此脚本
Develop_Mode = True # True means use computer camera. False means use dog camera
# global config
client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)
# creat a controller
controller = Controller(server_address)
Danger = DangerSignRecognition()
sign_detector = DangerSignRecognition()

has_arm = True
try:
    from utils.ArmController import ArmController
    arm_controller = ArmController("/dev/ttyUSB0")
    arm_controller.set_pose(1)
except Exception as e:
    print("no arm")
    has_arm = False

# start to exchange heartbeat pack
Heartbeat.send_heartbeat()



def warm_up(self):
    pack = struct.pack('<3i', 0x21010202, 0, 0)
    print(1)
    controller.send(pack)
    time.sleep(5)
    print(2)
    controller.send(pack)
    time.sleep(5)
    print(3)
    controller.send(pack)

    print("Waiting 15s......")
    time.sleep(15)
    if has_arm:
        arm_controller.set_pose(2)
    print("Rotating...")
    controller.send(struct.pack('<3i', 0x21010135, 13000, 0))
    time.sleep(3)  # need time to turn 360 degrees
    controller.send(struct.pack('<3i', 0x21010135, 0, 0))
    time.sleep(5)
    print(4)
    controller.send(pack)

warm_up()


if __name__ == '__main__':
    