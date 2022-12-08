import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(frame, (5,5),0)

    sobelx = cv2.Sobel(blurred_frame, cv2.CV_64F, 1,0)
    sobely = cv2.Sobel(blurred_frame, cv2.CV_64F, 0,1)

    laplacian = cv2.Laplacian(blurred_frame, cv2.CV_64F, ksize = 13)

    canny = cv2.Canny(blurred_frame,90,100,5)

    cv2.imshow("Image",frame)
    cv2.imshow("Sobelx",sobelx)
    cv2.imshow("Sobely",sobely)
    cv2.imshow("Laplacian",laplacian)
    cv2.imshow("canny",canny)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
