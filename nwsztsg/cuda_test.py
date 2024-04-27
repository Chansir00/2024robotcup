import time
import struct
import cv2 as cv
import numpy as np
import threading

from utils.DashboardRecognition import DashboardRecognition
from utils.Controller import Controller

Develop_Mode = True # True means use computer camera. False means use dog camera
client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)


global frame
global number
global is_update_number

if __name__ == '__main__':
    dashboard_detector = DashboardRecognition()
    controller = Controller(server_address)

    # try to use CUDA
    if cv.cuda.getCudaEnabledDeviceCount() != 0:
        backend = cv.dnn.DNN_BACKEND_CUDA
        target = cv.dnn.DNN_TARGET_CUDA
        print('done')
    else:
        backend = cv.dnn.DNN_BACKEND_DEFAULT
        target = cv.dnn.DNN_TARGET_CPU
        print('CUDA is not set, will fall back to CPU.')
        
        
