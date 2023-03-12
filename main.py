import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap =cv2.VideoCapture(0)
dector = HandDetector(maxHands=1)

offset =20
imgSize = 300
counter = 0

folder = "Data/B"


while True :
    succuss,img =cap.read()
    hands,img = dector.findHands(img)
    if hands:
        hands = hands[0]
        x, y, w,h =hands['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*225
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgCropShape = imgCrop.shape
        # imgWhite[0:imgCropShape[0],0:imgCropShape[1]] = imgCrop
        aspectRatio = h/w

        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            ImgResize = cv2.resize(imgCrop,(wCal,imgSize))
            ImgResizeShape = ImgResize.shape
            wGap = math.ceil((imgSize- wCal)/2)
            imgWhite[:, wGap:wCal +wGap] = ImgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            ImgResize = cv2.resize(imgCrop, (imgSize, hCal))
            ImgResizeShape = ImgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap,:] = ImgResize

        cv2.imshow('imageCrop', imgCrop)
        cv2.imshow('imageWhite', imgWhite)


    cv2.imshow('image', img)
    key = cv2.waitKey(4)
    if key == ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/imge_{time.time()} .jpg', imgWhite)
        print(counter)