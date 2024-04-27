import cv2
from pyzbar.pyzbar import decode
import numpy as np
class Read_QR:
    def __init__(self):

        pass

    def decode_qr(self, image):
        """识别图像中的所有QR码，并返回解码内容和图像"""
        qr_contents = []
        decoded_objects = decode(image)
        for obj in decoded_objects:
            data = obj.data.decode('utf-8')
            qr_contents.append(data)
            # 可选：绘制边界
            image = self.draw_qr_border(image, obj.polygon)
        return qr_contents, image

    def draw_qr_border(self, image, points):
        """在图像上绘制QR码边界"""
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        n = len(points)
        for j in range(n):
            cv2.line(image, points[j], points[(j+1) % n], (255,0,0), 3)
        return image

# 使用摄像头测试Read_QR类
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    qr_reader = Read_QR()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        qr_contents, processed_frame = qr_reader.decode_qr(frame)
        if qr_contents:
            print("Detected QR Code(s):", qr_contents)
        cv2.imshow("QR Code Scanner", processed_frame)
        if cv2.waitKey(1) == ord('q'):  # 按'q'退出
            break

    cap.release()
    cv2.destroyAllWindows()
