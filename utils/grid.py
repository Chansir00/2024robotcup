# from PIL import Image
# import matplotlib.pyplot as plt

# img = Image.open(r"X:\robot\2024\dangersign\collapse.png")

# plt.imshow(img)
# plt.axis('off')

# plt.grid(True,color='black',linewidth='0.5',linestyle='dotted')


# plt.show()

import cv2
img = cv2.imread(r"X:\robot\2024\dangersign\collapse.png")
gray_img = cv2.cvtColor(img,cv2.COLOR_BAYER_BGGR2GRAY)

grid_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('image', grid_img)
cv2.waitKey(0)