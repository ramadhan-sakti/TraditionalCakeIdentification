# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 22:49:24 2019

@author: adhan
"""
# =============================================================================
# Crop-Fit
# =============================================================================

import cv2
import numpy as np
import datetime

def crop(jajanan):
    startTime = datetime.datetime.now()
    for counter in range(1,181):
        img = cv2.imread('jajanan_segmented/'+jajanan+'/'+jajanan+'_segmented_'+str(counter)+'.jpg')
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        #Koordinat Objek
        obj = []
        koorX = []
        koorY = []
        for i in range(len(grayImg)):
            for j in range(len(grayImg[i])):
                if(grayImg[i][j] != 0):
                    obj.append([i,j])
        for i in obj:
            koorX.append(i[0])
            koorY.append(i[1])

        minX = min(koorX)
        minY = min(koorY)
        maxX = max(koorX)
        maxY = max(koorY)

        #Crop
        croppedImg = np.zeros(shape = (maxX-minX,maxY-minY,3))
        for i in range(len(croppedImg)):
            for j in range(len(croppedImg[i])):
                croppedImg[i][j][0] = img[i+minX][j+minY][0]
                croppedImg[i][j][1] = img[i+minX][j+minY][1]
                croppedImg[i][j][2] = img[i+minX][j+minY][2]
        croppedImg = croppedImg.astype(np.uint8)
        cv2.imwrite('jajanan_cropped/'+jajanan+'/'+jajanan+'_cropped_'+str(counter)+'.jpg', croppedImg)
    endTime = datetime.datetime.now()
    deltaTime = endTime-startTime
    print('Selesai Dalam => '+str(deltaTime))
    
crop('Ape')
crop('DadarGulung')
crop('Lumpur')
crop('PutuAyu')
crop('Soes')