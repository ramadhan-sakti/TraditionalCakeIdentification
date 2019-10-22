# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:41:18 2019

@author: adhan
"""

import cv2
import math
import numpy as np

# =============================================================================
# Init
# =============================================================================
img = cv2.imread('kue_lumpur_1.jpg')

cv2.imshow('lumpur',img)
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
#cv2.imshow('grayManual',grayImg)

# =============================================================================
# Convert RGB to XYZ(Untuk Lab)
# =============================================================================
xyzArr = np.zeros(shape = img.shape)

k = np.array([[0.49, 0.31, 0.20],
     [0.17697, 0.81240, 0.01063],
     [0.00, 0.01, 0.99]
     ])

xyz = np.zeros(3)

for i in range(len(img)):
    for j in range(len(img[i])):
        rgb = np.array([allR[i,j],allG[i,j],allG[i,j]])
        xyz = (1/0.17697) * k.dot(rgb)
        
        xyzArr[i][j][0] = xyz[0]
        xyzArr[i][j][1] = xyz[1]
        xyzArr[i][j][2] = xyz[2]

# =============================================================================
# Mengubah XYZ ke LAB
# =============================================================================
   
# Fungsi lab
def flab(q):
    if(q > 0.008856):
        return np.cbrt(q)
    else:
        return (7.787*q)+(16/116)

#Menggunakan Iluminasi D65
xn = 95.047
yn = 100.00
zn = 108.883       

l = 116 * flab(y/yn) - 16
a = 500 * (flab(x/xn) - flab(y/yn))
b = 200 * (flab(y/yn) - flab(z/zn))

# =============================================================================
# LBP
# =============================================================================
def getLBP(img):
    lbpLoop = np.arange(1,999)
    lbpBinarySeq = [0,0,0,0,0,0,0,0]
    lbpVal = 0
    lbpList = []

    for i in lbpLoop:
        for j in lbpLoop:
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

            for i in range (8): #Menghitung Nilai LBP Piksel [I,J]
                lbpVal += lbpBinarySeq[i]*math.pow(2,i)
                #print(lbpVal)
        
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
    
def getSkewness(channel,mean): ################################################Cek Lagi
    tempSkew = 0
    flatChannel = channel.ravel().tolist()
    for i in flatChannel:
        tempSkew += pow(i-mean,3)
    hasilSkew = round(math.pow((tempSkew/len(flatChannel)),1/3),2)
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