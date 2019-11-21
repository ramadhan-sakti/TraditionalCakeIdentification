# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:48:51 2019

@author: adhan
"""

import cv2
import math
import numpy as np

# =============================================================================
# Init
# =============================================================================
img = cv2.imread('kue_lumpur_1.jpg')

allB = img[:,:,0] #channel warna biru
allG = img[:,:,1] #channel warna hijau
allR = img[:,:,2] #channel warna merah

# =============================================================================
# OpenCV
# =============================================================================

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
xyzImg = cv2.cvtColor(img, cv2.COLOR_XYZ2BGR)
labImg = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

testRGB = [allR[0][0], allG[0][0], allB[0][0]]
testLab = [labImg[0][0][0], labImg[0][0][1], labImg[0][0][2]]
testXYZ = [xyzImg[0][0][0], xyzImg[0][0][1], xyzImg[0][0][2]]
# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================

#kd50 = np.array([[0.49, 0.31, 0.20],
#     [0.17697, 0.81240, 0.01063],
#     [0.00, 0.01, 0.99]
#     ])
def getXYZ(r,g,b):
    k = np.array([[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]
                  ])

    rgb = [120,55,89]
    rgbnorm = [rgb[0]/255, rgb[1]/255, rgb[2]/255]
    rgbnorm = [r/255,g/255,b/255]
    

    #xyz = (1/0.17697) * k.dot(rgbnorm)

    xyz = k.dot(rgbnorm)

    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    
    print(x,y,z)

# =============================================================================
# XYZ to RGB
# =============================================================================

k = np.array([[3.240479, -1.53715, -0.498535],
     [-0.969256, 1.875991, 0.041556],
     [0.055648, -0.204043, 1.057311]
     ])

xyznorm = [xyz[0], xyz[1], xyz[2]]

rgbt = k.dot(xyznorm)

rt = rgbt[0]
gt = rgbt[1]
bt = rgbt[2]
# =============================================================================
# Mengubah XYZ ke LAB
# =============================================================================
   
# Fungsi lab
def flab(q):
    if(q > 0.008856):
        return np.cbrt(q)
    else:
        return (7.787*q)+(16/116)

def getLab(x,y,z):

    #Menggunakan Iluminasi D65
    xn = 0.950456
    yn = 1
    zn = 1.088754
    
    if(y > 0.008856):
        l = 116 * np.cbrt(y) - 16
    else:
        l = 903.3 * y

    a = 500 * (flab(x/xn) - flab(y/yn))
    b = 200 * (flab(y/yn) - flab(z/zn))
    
    print(l,a,b)


#lnorm = (l*255)/100
#anorm = a+125
#bnorm = b+125

#
## Fungsi lab
#def flab(q):
#    if(q > 0.008856):
#        return np.cbrt(q)
#    else:
#        return (7.787*q)+(16/116)
#
##Menggunakan Iluminasi D65
#xn = 95.047
#yn = 100.00
#zn = 108.883       
#
#l = 116 * flab(y/yn) - 16
#a = 500 * (flab(x/xn) - flab(y/yn))
#b = 200 * (flab(y/yn) - flab(z/zn))


## Fungsi lab
#def flab(q):
#    if(q > 0.008856):
#        return np.cbrt(q)
#    else:
#        return (7.787*q)+(16/116)
#
##Menggunakan Iluminasi D65
#xn = 0.950456
#yn = 1
#zn = 1.088754      
#
#l = 116 * flab(y/yn) - 16
#a = 500 * (flab(x/xn) - flab(y/yn))
#b = 200 * (flab(y/yn) - flab(z/zn))