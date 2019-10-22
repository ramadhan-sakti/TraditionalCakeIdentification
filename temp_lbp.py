# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:00:08 2019

@author: adhan
"""
import numpy as np
import math

imgTest = np.array( [[8, 10, 0],
                       [8, 4, 2],
                       [8, 8, 6]
                       ])

lbpLoop = np.arange(1,999)
lbpBinarySeq = [0,0,0,0,0,0,0,0]
lbpVal = 0
i = 1
j = 1

#Atas-Kiri
if(imgTest[i,j] < imgTest[i-1,j-1]):
    lbpBinarySeq[0] = 1
else:
    lbpBinarySeq[0] = 0
#Atas
if(imgTest[i,j] < imgTest[i-1,j]):
    lbpBinarySeq[1] = 1
else:
    lbpBinarySeq[1] = 0
#Atas-Kanan
if(imgTest[i,j] < imgTest[i-1,j+1]):
    lbpBinarySeq[2] = 1
else:
    lbpBinarySeq[2] = 0
#Tengah-Kiri
if(imgTest[i,j] < imgTest[i,j-1]):
    lbpBinarySeq[3] = 1
else:
    lbpBinarySeq[3] = 0
#Tengah-Kanan
if(imgTest[i,j] < imgTest[i,j+1]):
    lbpBinarySeq[4] = 1
else:
    lbpBinarySeq[4] = 0
#Bawah-Kiri
if(imgTest[i,j] < imgTest[i+1,j-1]):
    lbpBinarySeq[5] = 1
else:
    lbpBinarySeq[5] = 0
#Bawah
if(imgTest[i,j] < imgTest[i+1,j]):
    lbpBinarySeq[6] = 1
else:
    lbpBinarySeq[6] = 0
#Bawah-Kanan
if(imgTest[i,j] < imgTest[i+1,j+1]):
    lbpBinarySeq[7] = 1
else:
    lbpBinarySeq[7] = 0


for i in range (8):
    lbpVal += lbpBinarySeq[i]*math.pow(2,i)
    print(lbpVal)

#OK