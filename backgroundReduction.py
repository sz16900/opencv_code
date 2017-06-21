import cv2
import numpy as np


cap = cv2.VideoCapture('cut.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()


while(cap.isOpened()):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow('original', frame)
    cv2.imshow('fg', fgmask)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
