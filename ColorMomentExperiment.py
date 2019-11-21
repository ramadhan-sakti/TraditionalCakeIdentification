# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:41:18 2019

@author: adhan
"""
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

# =============================================================================
# Init
# =============================================================================
#img = cv2.imread('lumpur_2.jpg')
img = cv2.imread('rama.jpg')
#img = cv2.imread('red.jpg')
cv2.imshow('lumpur',img)
allB = img[:,:,0] #channel warna biru
allG = img[:,:,1] #channel warna hijau
allR = img[:,:,2] #channel warna merah


# =============================================================================
# Grayscale
# =============================================================================
grayImg = np.zeros(shape = (1000,1000))
for i in range(len(img)):
    for j in range(len(img[i])):
        grayImg[i][j] = (0.333*allR[i][j])+(0.5*allG[i][j]+(0.1666*allB[i][j]))

grayImg = grayImg.astype(np.uint8)

# =============================================================================
# Convert RGB to XYZ(Untuk Lab)
# =============================================================================
def toXYZ(img):
    xyzArr = np.zeros(shape = img.shape)

    #k = np.array([[0.49, 0.31, 0.20],
    #     [0.17697, 0.81240, 0.01063],
    #     [0.00, 0.01, 0.99]
    #     ])
    
    k = np.array([[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]
                  ])

    for i in range(len(img)):
        for j in range(len(img[i])):
            rgbnorm = np.array([allR[i,j]/255,allG[i,j]/255,allG[i,j]/255])
            #xyz = (1/0.17697) * k.dot(rgb)
            xyz = k.dot(rgbnorm)
            
            xyzArr[i][j][0] = xyz[0]
            xyzArr[i][j][1] = xyz[1]
            xyzArr[i][j][2] = xyz[2]
    
    return xyzArr

# =============================================================================
# Mengubah XYZ ke LAB
# =============================================================================
# Fungsi lab
def flab(q):
    if(q > 0.008856):
        return np.cbrt(q)
    else:
        return (7.787*q)+(16/116)

def toLab(img): #gambar input adalah citra ruang warna XYZ
    labArr = np.zeros(shape = img.shape)
    #Menggunakan Iluminasi D65
    xn = 0.950456
    yn = 1
    zn = 1.088754
    
        
    for i in range(len(img)):
        for j in range(len(img[i])):
            x = img[i][j][0]
            y = img[i][j][1]
            z = img[i][j][2]
            
            if(y > 0.008856):
                l = 116 * np.cbrt(y) - 16
            else:
                l = 903.3 * y
            
            a = 500 * (flab(x/xn) - flab(y/yn))
            b = 200 * (flab(y/yn) - flab(z/zn))
            
            labArr[i][j][0] = l
            labArr[i][j][1] = a
            labArr[i][j][2] = b
    
    return labArr
# =============================================================================
# LBP
# =============================================================================
def getLBP(grayImg):
    lbpBinarySeq = [0,0,0,0,0,0,0,0]
    lbpVal = 0
    lbpList = []
    lbpImg = np.zeros(shape = grayImg.shape)
    
    for i in range(1,len(grayImg)-1):
        for j in range(1,len(grayImg)-1):
            #Atas-Kiri
            if(grayImg[i,j] < grayImg[i-1,j-1]):
                lbpBinarySeq[0] = 1
            else:
                lbpBinarySeq[0] = 0
            #Atas
            if(grayImg[i,j] < grayImg[i-1,j]):
                lbpBinarySeq[1] = 1
            else:
                lbpBinarySeq[1] = 0
            #Atas-Kanan
            if(grayImg[i,j] < grayImg[i-1,j+1]):
                lbpBinarySeq[2] = 1
            else:
                lbpBinarySeq[2] = 0
            #Tengah-Kiri
            if(grayImg[i,j] < grayImg[i,j-1]):
                lbpBinarySeq[3] = 1
            else:
                lbpBinarySeq[3] = 0
            #Tengah-Kanan
            if(grayImg[i,j] < grayImg[i,j+1]):
                lbpBinarySeq[4] = 1
            else:
                lbpBinarySeq[4] = 0
            #Bawah-Kiri
            if(grayImg[i,j] < grayImg[i+1,j-1]):
                lbpBinarySeq[5] = 1
            else:
                lbpBinarySeq[5] = 0
            #Bawah
            if(grayImg[i,j] < grayImg[i+1,j]):
                lbpBinarySeq[6] = 1
            else:
                lbpBinarySeq[6] = 0
            #Bawah-Kanan
            if(grayImg[i,j] < grayImg[i+1,j+1]):
                lbpBinarySeq[7] = 1
            else:
                lbpBinarySeq[7] = 0

            for y in range (8): #Menghitung Nilai LBP Piksel [I,J]
                lbpVal += lbpBinarySeq[y]*math.pow(2,y)
                #print(lbpVal)
            
            lbpImg[i][j] = lbpVal
            
            lbpList.append(lbpVal) #Simpan nilai2 LBP Citra
            lbpBinarySeq = [0,0,0,0,0,0,0,0]
            lbpVal = 0 #Reset nilai LBP utk pixel selanjutnya
        
    lbpList = np.array(lbpList)
    return lbpList
        
# =============================================================================
# Convert to HSV
# =============================================================================

def toHSV(img):
    hsvArr = np.zeros(shape = img.shape)

    for i in range(len(img)):
        for j in range(len(img[i])):
            r = float(img[i][j][2])
            g = float(img[i][j][1])
            b = float(img[i][j][0])
        
            #Menghitung V
            v = max(r,g,b)
            vm = v-min(r,g,b)
    
            #Menghitung S
            if(v == 0):
                s = 0
            elif(v > 0):
                s = round(vm/v,2)
        
            #Menghitung H
            if(s == 0):
                h = 0
            elif(v == r):
                h = round(60*(((g-b)/vm)%6),2)
            elif(v == g):
                h = round(60*(2+((b-r)/vm)),2)
            elif(v == b):
                h = round(60*(4+((r-g)/vm)),2)
        
            #normalisasi V
            v = round(v/255,2)
            
            #Assign
            hsvArr[i][j][0] = h
            hsvArr[i][j][1] = s
            hsvArr[i][j][2] = v

    return hsvArr
# =============================================================================

# =============================================================================
# Hitung Mean
# =============================================================================

def getMean(channel):
    temp = 0
    flatChannel = channel.ravel().tolist()
    for i in flatChannel:
        temp += i
        hasilMean = temp/len(flatChannel)
    return hasilMean
# =============================================================================

# =============================================================================
# Hitung Standar Deviasi
# =============================================================================

def getSTD(channel, mean):
    tempSTD = 0
    flatChannel = channel.ravel().tolist()
    for i in flatChannel:
        tempSTD += pow(i-mean,2)
    hasilSTD = round(math.sqrt(tempSTD/len(flatChannel)),2)
    return hasilSTD
# =============================================================================

# =============================================================================
# Hitung Skewness
# =============================================================================
    
def getSkewness(channel,mean):
    tempSkew = 0
    flatChannel = channel.ravel().tolist()
    for i in flatChannel:
        tempSkew += pow(i-mean,3)
    hasilSkew = round(np.cbrt(tempSkew/len(flatChannel)),2)
    return hasilSkew
# =============================================================================

# =============================================================================
# Test Mean, STD, Skewness
# =============================================================================
meanR = getMean(allR)
stdR = getSTD(allR, meanR)
skewR = getSkewness(allR, meanR)


# =============================================================================
# Test HSV result
# =============================================================================

hsvArr = toHSV(img)

tr = img[0][1][2]
tg = img[0][1][1]
tb = img[0][1][0]

th = hsvArr[0][1][0]
ts = hsvArr[0][1][1]
tv = hsvArr[0][1][2]

print('R= ',tr,'G= ',tg,'B= ',tb)
print('H= ',th,'S= ',ts,'V=',tv)