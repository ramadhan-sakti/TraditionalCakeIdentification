# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 08:58:19 2019
"""
#https://stackoverflow.com/questions/26392336/importing-images-from-a-directory-python
#https://www.analyticsvidhya.com/blog/2019/08/3-techniques-extract-features-from-image-data-machine-learning-python/
#https://www.codementor.io/isaib.cicourel/image-manipulation-in-python-du1089j1u


import cv2
cv2.__version__
import numpy as np
from PIL import I

imageJajan = cv2.imread('kue_lumpur_1.jpg')
imageJajanRGB = cv2.cvtColor(imageJajan,cv2.COLOR_BGR2RGB)
imageHSV = cv2.cvtColor(imageJajanRGB, cv2.COLOR_RGB2HSV)
im = Image.open('kue_lumpur_1.jpg')
im = im.convert('RGB')

imHSV = im.convert('HSV')
dataHSV = np.array(imHSV)
hue,saturation, value = dataHSV.T

data = np.array(im)   # "data" is a height x width x 4 numpy array
red, green, blue = data.T # Temporarily unpack the bands for readability

pixels = im.load()

#for i in range(im.size[0]):
#        for j in range(im.size[1]):
#            if pixels[i,j] == (199, 198, 194):
#                pixels[i,j] = (0, 0 ,0)
                
#im.show()

def rgb2hsv(r,g,b):
    R = r[:,:]/255
    G = g[:,:]/255
    B = b[:,:]/255

    vMaxR = np.amax(R)
    vMinR = np.amin(R)
    vMaxG = np.amax(G)
    vMinG = np.amin(G)
    vMaxB = np.amax(B)
    vMinB = np.amin(B)
    
    #Nyari Value
    V = max(vMaxR, vMaxG, vMaxB)
    Vmin = V - min(vMinR, vMinG, vMinB)
    
    #Nyari Saturatuion
    if(V == 0):
        s = 0
    elif(V > 0):
        s = (Vmin/V) * 100
    
    for i in range(im.size[0]):
        for i in range(im.size[1]):
            if(V == Vmin):
                h = 0
            elif(V == R):
                for i in range(r):
                    h = (60 * ((G-B)/Vmin) + 360) % 360
            elif(V == G):
                for i in range(G):
                    h = (60 * ((B-R)/Vmin) + 120) % 360
            elif(V == B):
                for i in range(B):
                    h = (60 * ((R-G)/Vmin) + 240) % 360
            print(h,s,V)
        
rgb2hsv(red,green,blue)
            
            
            
im2 = Image.fromarray(data)
im2.show()
