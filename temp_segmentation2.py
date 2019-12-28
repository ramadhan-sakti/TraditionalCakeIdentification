# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:15:47 2019

@author: adhan
"""

import cv2
import numpy as np

# =============================================================================
# Grayscale
# =============================================================================
def toGray(img):
    grayImg = np.zeros(shape = (1000,1000))
    for i in range(len(img)):
        for j in range(len(img[i])):
            grayImg[i][j] = (0.299*allR[i][j])+(0.587*allG[i][j]+(0.144*allB[i][j]))

    grayImg = grayImg.astype(np.uint8)
    return grayImg
# =============================================================================
# Binerisasi
# =============================================================================
def binerisasi(grayImg,tres):
    biner = np.zeros(shape = grayImg.shape)
    for i in range(len(grayImg)):
        for j in range(len(grayImg[i])):
            if(grayImg[i][j] <= tres):
                biner[i][j] = 0
            else:
                biner[i][j] = 255
    biner = biner.astype(np.uint8)
    return biner
# =============================================================================
# Mask
# =============================================================================
def mask(img_fix):
    imgHasil = img.copy()
    for i in range(len(img_fix)):
            for j in range(len(img_fix[i])):
                if(img_fix[i][j] == 0):
                    imgHasil[i][j] = 0
    return imgHasil
# =============================================================================
#                                     Test
# =============================================================================

# =============================================================================
#                                 Inisialisasi     X
# =============================================================================
img = cv2.imread('jajanan_resized/PutuAyu/PutuAyu_resized_180.jpg')

allB = img[:,:,0] #channel warna biru
allG = img[:,:,1] #channel warna hijau
allR = img[:,:,2] #channel warna merah
kernel = np.ones((5,5), np.uint8)
kernelLPF = np.ones((5,5),np.float32)/25
cv2.imshow('Warna',img)

#Gray
#grayImgA = toGray(img)
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Skala Abu',grayImg)

#LPF
grayImg = cv2.filter2D(grayImg,-1,kernelLPF)

#Biner
biner = binerisasi(grayImg,65)
cv2.imshow('Gambar Biner LPF',biner)

# =============================================================================
#                                 EROSI-DILASI
# =============================================================================
# Dilasi Erosi
img_erosion = cv2.erode(biner, kernel, iterations = 1)
img_dilation = cv2.dilate(biner, kernel, iterations = 15)

img_erosion = cv2.erode(img_dilation, kernel, iterations = 15)
img_dilation = cv2.dilate(img_erosion, kernel, iterations = 15)

#Show Erosi
cv2.imshow('Erosi',img_erosion)

#Show Dilasi
cv2.imshow('Dilasi',img_dilation)

#Fix Erosi
img_fix = img_erosion

#Fix Dilasi
img_fix = img_dilation

# =============================================================================
#                                   MASK
# =============================================================================
imgHasil = mask(img_fix)
cv2.imshow('Hasil Akhir',imgHasil)
# =============================================================================
#                                   SAVE          X
# =============================================================================
cv2.imwrite('jajanan_segmented/PutuAyu/PutuAyu_segmented_180.jpg', imgHasil)