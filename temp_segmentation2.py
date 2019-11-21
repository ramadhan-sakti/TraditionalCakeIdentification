# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:15:47 2019

@author: adhan
"""

import cv2
import math
import numpy as np

#img = cv2.imread('kue_lumpur_1.jpg')
img = cv2.imread('lumpur_2.jpg')

allB = img[:,:,0] #channel warna biru
allG = img[:,:,1] #channel warna hijau
allR = img[:,:,2] #channel warna merah



# =============================================================================
# Grayscale
# =============================================================================
#grayImgOpenCV = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayImg = np.zeros(shape = (1000,1000))
for i in range(len(img)):
    for j in range(len(img[i])):
        grayImg[i][j] = (0.333*allR[i][j])+(0.5*allG[i][j]+(0.1666*allB[i][j]))

grayImg = grayImg.astype(np.uint8)
cv2.imshow('grayManual',grayImg)

# =============================================================================
# Binerisasi
# =============================================================================
#Static Global thresh
#ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#
#cv2.imshow('grayManual2',th1)

def binerisasi(grayImg,tres):
    biner = np.zeros(shape = grayImg.shape)
    for i in range(len(grayImg)):
        for j in range(len(grayImg[i])):
            if(grayImg[i][j] <= tres):
                biner[i][j] = 0
            else:
                biner[i][j] = 255
    #biner = biner.astype(np.uint8)
    cv2.imshow('biner',biner)
    return biner

biner = binerisasi(grayImg,210)
# =============================================================================
# Dilasi Erosi
# =============================================================================

kernel = np.ones((5,5), np.uint8) 
  
img_erosion = cv2.erode(biner, kernel, iterations=3)
img_dilation = cv2.dilate(biner, kernel, iterations = 2)

img_dilation = cv2.dilate(img_erosion, kernel, iterations=15) 
img_erosion = cv2.dilate(img_dilation, kernel, iterations=1) 

cv2.imshow('biner',img_erosion)
cv2.imshow('biner',img_dilation)
cv2.imshow('biner',biner)

img_fix = img_dilation
cv2.imshow('fix',img_fix)
# =============================================================================
# Mask
# =============================================================================
#simpan koordinat obj
coordinate = []
for i in range(len(img_fix)):
        for j in range(len(img_fix[i])):
            if(img_fix[i][j] != 0):
                coordinate.append([i,j])

#imgPutih = np.zeros(shape = img.shape)
imgPutih = img.copy()
#putihkan background
for i in coordinate:
    imgPutih[i[0]][i[1]] = 255

imgPutih = imgPutih.astype(np.uint8)

cv2.imshow('hasil',imgPutih)
cv2.imshow('img',img)