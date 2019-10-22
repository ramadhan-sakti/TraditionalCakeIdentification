# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:43:51 2019

@author: adhan
"""

import math
# =============================================================================
# #MEAN
# =============================================================================
channel = [1,2,3,9]

temp = 0
for i in channel:
    temp += i
mean = temp/len(channel)
print(mean)

# =============================================================================
# STDev
# =============================================================================

tempSTD = 0
for i in channel:
    tempSTD += pow(i-mean,2)
hasilSTD = round(math.sqrt(tempSTD/len(channel)),2)

print(hasilSTD) #test

# =============================================================================
# Skewness
# =============================================================================

tempSkew = 0
for i in channel:
    tempSkew += pow((i-mean),3)
hasilSkew = round(math.pow((tempSkew/len(channel)),1/3),2)

print(hasilSkew)