# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:16:01 2019

@author: adhan
"""
import cv2
cv2.__version__
import numpy as np
#from PIL import I

img = cv2.imread('kue_lumpur_1.jpg')

#print(lumpur)
#cv2.imshow('lumpur',lumpur)

allB = img[:,:,0] #channel warna biru
allG = img[:,:,1] #channel warna hijau
allR = img[:,:,2] #channel warna merah

newArr = np.ndarray(shape = img.shape)

# =============================================================================
# Test RGB-HSV Conversion
# =============================================================================

r = float(img[0][0][2])
g = float(img[0][0][1])
b = float(img[0][0][0])

#Menghitung V
v = max(r,g,b)
vm = v-min(r,g,b)
    
#Menghitung S
if(v == 0):
   s = 0
elif(v > 0):
       s = vm/v
       s = round(s,2)
       
#Menghitung H
if(s == 0):
    h = 0
elif(v == r):
    h = 60*(((g-b)/vm)%6)
    h = round(h,2)
elif(v == g):
    h = 60*(2+((b-r)/vm))
    h = round(h,2)
elif(v == b):
    h = 60*(4+((r-g)/vm))
    h = round(h,2)

#normalisasi v
v = round(v/255,2)

# =============================================================================
# Print RGB and HSV
# =============================================================================
print('r = ',r)
print('g = ',g)
print('b = ',b)
print('##########################')
print('h = ',h)
print('s = ',s)
print('v = ',v)