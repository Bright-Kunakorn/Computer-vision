import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("windows.jpg",)

# converting to gray scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#blur image to reduce noise with 3x3 kernel size and set 0 to x and y parameter of GaussianBlur
img = cv2.GaussianBlur(img, (3,3),0)

#cv2.CV_64F set data type to double 64  bits
sobelx = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize = 5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0,1, ksize = 5)

laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize = 3)

canny = cv2.Canny(img,100,150)

plt.subplot(2,2,1),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
