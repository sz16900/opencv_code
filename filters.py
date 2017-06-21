import cv2
import numpy as np
from scipy import signal
from scipy import misc
from scipy.signal import argrelextrema

cap = cv2.VideoCapture('cut.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((5,5),np.float32)/25

scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

# # Dont know yet
# X = list(len(cap))
# Y = list(len(cap))

while(cap.isOpened()):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    # smooth the image
    gaussian = cv2.filter2D(fgmask,-1,kernel)
    laplacian = cv2.Laplacian(fgmask,cv2.CV_64F)
    blob_img = signal.convolve2d(fgmask, scharr, boundary='symm', mode='same')
    # For now, this is like this.. gotta change it
    inds = np.where(blob_img < 0.7)
    blob_img[inds] = np.nan
    imax = argrelextrema(blob_img, np.greater)
    # sobelx = cv2.Sobel(fgmask,cv2.CV_64F,1,0,ksize=5)
    # sobely = cv2.Sobel(fgmask,cv2.CV_64F,0,1,ksize=5)

    cv2.imshow('original', frame)
    cv2.imshow('fg', fgmask)
    cv2.imshow('gaussian', gaussian)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('blob', np.absolute(blob_img))
    # cv2.imshow('x', sobelx)
    # cv2.imshow('y', sobely)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
