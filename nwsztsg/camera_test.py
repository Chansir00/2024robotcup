import cv2

cap_1 = cv2.VideoCapture("/dev/video3", cv2.CAP_V4L2)
cap_2 = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)
# cap_1 = cv2.VideoCapture(3, cv2.CAP_V4L2)
# cap_2 = cv2.VideoCapture(4, cv2.CAP_V4L2)

# cap_1 = cv2.VideoCapture(3)
# cap_2 = cv2.VideoCapture(4)
while(1):
    # get a frame
    ret_1, frame_1 = cap_1.read()
    ret_2, frame_2 = cap_2.read()
    cv2.imshow("cam_3", frame_1)
    cv2.imshow("cam_4", frame_2)
    # 等待按键事件发生 等待1ms
    key = cv2.waitKey(1)
    if key == ord('q'):
        # 如果按键为q 代表quit 退出程序
        print("程序正常退出..")
        break

cap_1.release()
cap_2.release()
cv2.destroyAllWindows()
    

