# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:54:51 2019

@author: adhan
"""

import numpy as np
import cv2
import os

NumKue = 1
jajan = 'Ape'
# =============================================================================
# Start
# =============================================================================

for filename in os.listdir('jajanan_black/'+jajan):
    print(filename)
    img = cv2.imread('jajanan_black/'+jajan+'/'+filename)
    constant = cv2.copyMakeBorder(img,left = 0, right = 0,
                              top = 500, bottom = 500,
                              borderType = cv2.BORDER_CONSTANT,
                              value = [0,0,0])
    dim = (1000,1000)
    resized = cv2.resize(constant, dim, interpolation = cv2.INTER_NEAREST)
    cv2.imwrite('jajanan_resized/'+jajan+'/'+jajan+'_resized_'+str(NumKue)+'.jpg', resized)
    NumKue+=1
Numkue = 1