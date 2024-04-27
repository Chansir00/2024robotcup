import cv2
ID = 0
while(1):
    cap = cv2.VideoCapture(ID)
    # get a frame
    ret, frame = cap.read()
    if ret == False:
        ID += 1
    else:
        print(ID)
        if ID >= 1000:
            break

