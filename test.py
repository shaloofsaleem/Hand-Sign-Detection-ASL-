import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from  cvzone.ClassificationModule import Classifier

cap =cv2.VideoCapture(0)

dector = HandDetector(maxHands=1)
classifier = Classifier("Model/converted_keras/keras_model.h5", "Model/converted_keras/labels.txt")

offset =20
imgSize = 300
counter = 0
labels = ['A','B','C']

folder = "Data/B"


while True :
    succuss,img =cap.read()
    imgOutput = img.copy()
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
            pridction, index = classifier.getPrediction(imgWhite, draw=False)
            print(pridction,index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            ImgResize = cv2.resize(imgCrop, (imgSize, hCal))
            ImgResizeShape = ImgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap,:] = ImgResize
            pridction, index = classifier.getPrediction(imgWhite, draw=False)
        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x - offset+90,y - offset-50+50), (255, 0, 255), cv2.FILLED)

        cv2.putText(imgOutput,labels[index], (x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,225,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset), (255,0,255), 4)

        cv2.imshow('imageCrop', imgCrop)
        cv2.imshow('imageWhite', imgWhite)


    cv2.imshow('image', imgOutput)
    cv2.waitKey(4)
