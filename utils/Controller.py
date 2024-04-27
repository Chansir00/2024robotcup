import socket
import struct
import threading
import time

class Controller:
    def __init__(self, dst):
        self.lock = False
        self.last_ges = "stop"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = dst
        self.count_handpose = 0

    # used to send a pack to robot dog
    def send(self, pack):
        self.socket.sendto(pack, self.dst)

    def drive_dog(self, ges, val = 10000):
        print(ges)
        if self.count_handpose >3:
            return
        if self.lock or ges == self.last_ges:
            return

        if self.last_ges == "squat" and ges != "squat":
            # stand up
            self.send(struct.pack('<3i', 0x21010202, 0, 0))
            self.count_handpose+1
        else:
            # stop all actions
            self.send(struct.pack('<3i', 0x21010130, 0, 0))
            self.send(struct.pack('<3i', 0x21010131, 0, 0))
            self.send(struct.pack('<3i', 0x21010102, 0, 0))
            self.send(struct.pack('<3i', 0x21010135, 0, 0))
            self.count_handpose+1

        if self.last_ges != "squat" and ges == "squat":
            self.send(struct.pack('<3i', 0x21010202, 0, 0))
            self.count_handpose+1
        elif ges == "turn_left":
            def turn_360():
                self.lock = True
                self.send(struct.pack('<3i', 0x21010135, 13000, 0))
                time.sleep(2.84)  # need time to turn 360 degrees
                self.send(struct.pack('<3i', 0x21010135, 0, 0))
                self.lock = False

            turn_thread = threading.Thread(target=turn_360)
            turn_thread.start()
            self.count_handpose+1
        elif ges == "turn_right":
            def turn_360():
                self.lock = True
                self.send(struct.pack('<3i', 0x21010135, -13000, 0))
                time.sleep(2.84)  # need time to turn 360 degrees
                self.send(struct.pack('<3i', 0x21010135, 0, 0))
                self.lock = False

            turn_thread = threading.Thread(target=turn_360)
            turn_thread.start()
            self.count_handpose+1
        elif ges == "twisting":
            def dance():
                self.lock = True
                self.send(struct.pack('<3i', 0x21010204, 0, 0))
                time.sleep(5)  # dance 22 seconds then it can process next gesture
                self.lock = False

            twist_thread = threading.Thread(target=dance)
            twist_thread.start()
            self.count_handpose+1
        elif ges == "forward":
            self.send(struct.pack('<3i', 0x21010130, val, 0))
            time.sleep(1)
            self.send(struct.pack('<3i', 0x21010130, 0, 0))
            self.count_handpose+1
        elif ges == "back":
            self.send(struct.pack('<3i', 0x21010130, -val, 0))
            time.sleep(1)
            self.send(struct.pack('<3i', 0x21010130, 0, 0))
            self.count_handpose+1
        elif ges == "right":
            self.send(struct.pack('<3i', 0x21010131, val, 0))
            time.sleep(1)
            self.send(struct.pack('<3i', 0x21010131, 0, 0))
            self.count_handpose+1
        elif ges == "left":
            self.send(struct.pack('<3i', 0x21010131, -val, 0))
            time.sleep(1)
            self.send(struct.pack('<3i', 0x21010131, 0, 0))
            self.count_handpose+1

        self.last_ges = ges

    def stand_up(self):
        self.send(0x21010202,0,0)
        self.send(0x21010D06,0,0)
        time.sleep(2)

    def forward(self,duration):
        start_time = time.time()
        while(time.time()-start_time < duration):
            self.send(0x21010130,13000,0)
            time.sleep(0.05)

    def back(self,duration):
        start_time = time.time()
        while(time.time()-start_time < duration):
            self.send(0x21010130,-13000,0)
            time.sleep(0.05)
    

    def move_right(self,duration):
        start_time = time.time()
        while(time.time()-start_time < duration):
            self.send(0x21010131,-13000,0)
            time.sleep(0.05)


    def move_left(self,duration):
        start_time = time.time()
        while(time.time()-start_time < duration):
            self.send(0x21010131,13000,0)
            time.sleep(0.05)

    def turn_left(self,angle):  #>\9553\
        self.send(struct.pack('<3i', 0x21010135, angle, 0))
        time.sleep(2)

    def turn_right(self,angle):
        self.send(struct.pack('<3i', 0x21010135, angle, 0))
        time.sleep(2)

    def nod(self):
        self.send('<3i', 0x21010309, 0, 0)
        time.sleep(2)
