# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:00:08 2019

@author: adhan
"""
import numpy as np
import math

grayImg = np.array([[164, 30, 99, 120, 210, 130], 
                    [200, 150, 177, 180, 120, 21], 
                    [83, 215, 20, 45, 110, 179], 
                    [80, 220, 166, 62, 30, 139],
                    [203, 201, 150, 160, 170, 120],
                    [210, 160, 39, 49, 51, 240]])
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
        
        print("[",i,",",j,"]")
        lbpImg[i][j] = lbpVal
            
        lbpList.append(lbpVal) #Simpan nilai2 LBP Citra
        lbpBinarySeq = [0,0,0,0,0,0,0,0]
        lbpVal = 0 #Reset nilai LBP utk pixel selanjutnya
        
lbpList = np.array(lbpList)
