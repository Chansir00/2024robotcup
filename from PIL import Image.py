from PIL import Image
import matplotlib.pyplot as plt



img=Image.open("X:/robot/2024/dangersign/1(1).jpg")

plt.imshow(img)
plt.axis('off')

plt.grid(True,color='black',linewidth='1',linestyle='dotted')


plt.show()