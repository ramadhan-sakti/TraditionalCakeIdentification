# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:45:23 2019

@author: adhan
"""

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('kue_lumpur_1.jpg',0)

#Static Global thresh
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#Otsu tresh
ret2, th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Otsu + Gaussian Blur thresh
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#plot gambar & histogram

images = [img,0,th1,
          img,0,th2,
          blur,0,th3
          ]

titles = ['Lumpur asli','Histogram','Treshold global statis',
          'Lumpur asli','Histogram','Otsu Treshold',
          'Otsu + Gauss','Histogram','Otsu + Gauss',
         ]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])
plt.show()