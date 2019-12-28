# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:41:18 2019

@author: adhan
"""
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
import datetime

namaKolom = ['Nama Item','meanR', 'meanG', 'meanB', 'meanH', 'meanS', 'meanV', 'meanlabL', 
             'meanlabA', 'meanlabB', 'meanLBP', 'stdR', 'stdG', 'stdB', 'stdH', 'stdS', 'stdV', 'stdlabL', 
             'stdlabA', 'stdlabB', 'stdLBP', 'skewR', 'skewG', 'skewB', 'skewH', 'skewS', 'skewV', 'skewlabL', 
             'skewlabA', 'skewlabB', 'skewLBP', 'Class']
# =============================================================================
# Grayscale
# =============================================================================
def toGray(img):
    grayImg = np.zeros(shape = (img.shape[0],img.shape[1]))
    for i in range(len(img)):
        for j in range(len(img[i])):
            grayImg[i][j] = (0.333*img[i][j][2])+(0.5*img[i][j][1]+(0.1666*img[i][j][0]))

    grayImg = grayImg.astype(np.uint8)
    return grayImg

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
            rgbnorm = np.array([img[i][j][2]/255,img[i][j][1]/255,img[i][j][0]/255])
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
        for j in range(1,len(grayImg[i])-1):
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
    
    lbpImg = lbpImg.astype(np.uint8)
    #cv2.imshow('gambarLBP',lbpImg)
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
# Get All Features
# =============================================================================

def getFeature(jajanan,label): #Menerima argumen string nama jajanan
    featureSet = pd.DataFrame(columns = namaKolom)
    startTime = datetime.datetime.now()
    for i in range(1,181):
    #read jajanan
        img = cv2.imread('toBeExtracted/'+jajanan+'_cropped_'+str(i)+'.jpg')
        print('Citra '+jajanan+' ke-'+str(i))
# =============================================================================
# Get Mean, STD, Skewness dari seluruh channel
# =============================================================================
        allR = img[:,:,2]
        allG = img[:,:,1]
        allB = img[:,:,0]
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsvImg = toHSV(img)
        xyzImg = toXYZ(img)
        labImg = toLab(xyzImg)
        fiturLBP = getLBP(grayImg)
    
        #Fitur Mean
        meanR = getMean(allR)
        meanG = getMean(allG)
        meanB = getMean(allB)
        
        meanH = getMean(hsvImg[:,:,0])
        meanS = getMean(hsvImg[:,:,1])
        meanV = getMean(hsvImg[:,:,2])
        
        meanlabL = getMean(labImg[:,:,0])
        meanlabA = getMean(labImg[:,:,1])
        meanlabB = getMean(labImg[:,:,2])
        meanLBP = getMean(fiturLBP)

        #Fitur STD
        stdR = getSTD(allR, meanR)
        stdG = getSTD(allG, meanG)
        stdB = getSTD(allB, meanB)
        
        stdH = getSTD(hsvImg[:,:,0], meanH)
        stdS = getSTD(hsvImg[:,:,1], meanS) 
        stdV = getSTD(hsvImg[:,:,2], meanV)
        
        stdlabL = getSTD(labImg[:,:,0], meanlabL)
        stdlabA = getSTD(labImg[:,:,1], meanlabA)
        stdlabB = getSTD(labImg[:,:,2], meanlabB)
        
        stdLBP = getSTD(fiturLBP, meanLBP)
        
        #Fitur Skewness
        skewR = getSkewness(allR, meanR)
        skewG = getSkewness(allG, meanG)
        skewB = getSkewness(allB, meanB)
    
        skewH = getSkewness(hsvImg[:,:,0], meanH)
        skewS = getSkewness(hsvImg[:,:,1], meanS)
        skewV = getSkewness(hsvImg[:,:,2], meanV)
        
        skewlabL = getSkewness(labImg[:,:,0], meanlabL)
        skewlabA = getSkewness(labImg[:,:,1], meanlabA)
        skewlabB = getSkewness(labImg[:,:,2], meanlabB)
        
        skewLBP = getSkewness(fiturLBP, meanLBP)
    
        ## Save as cvs
    
        featureSet.loc[i] = [jajanan+'_'+str(i),meanR, meanG, meanB, meanH, meanS, meanV, meanlabL, 
                     meanlabA, meanlabB, meanLBP, stdR, stdG, stdB, stdH, stdS, stdV, 
                     stdlabL, stdlabA, stdlabB, stdLBP, skewR, skewG, skewB, skewH, skewS, 
                     skewV, skewlabL, skewlabA, skewlabB, skewLBP, label]

    #Save to excel
    featureSet.to_excel('FeatureSet_'+jajanan+'.xlsx')
    endTime = datetime.datetime.now()
    deltaTime = endTime-startTime
    print('Selesai Dalam => '+str(deltaTime))

# =============================================================================
# Run
# =============================================================================
getFeature('Ape',1)
#getFeature('DadarGulung',2)
getFeature('Lumpur',3)
getFeature('PutuAyu',4)
getFeature('Soes',5)