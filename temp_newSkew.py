# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:56:45 2019

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

def getMean(channel):
    temp = 0
    flatChannel = channel.ravel().tolist()
    for i in flatChannel:
        temp += i
    hasilMean = temp/len(flatChannel)
    return hasilMean

def getSTD(channel, mean):
    tempSTD = 0
    flatChannel = channel.ravel().tolist()
    for i in flatChannel:
        tempSTD += pow(i-mean,2)
    hasilSTD = round(math.sqrt(tempSTD/len(flatChannel)),2)
    return hasilSTD

def getVar(channel, mean):
    tempVar = 0
    flatChannel = channel.ravel().tolist()
    for i in flatChannel:
        tempVar += pow(i-mean,2)
    hasilVar = round(tempVar/len(flatChannel),2)
    return hasilVar

#Stricker
def getSkewStricker(channel,mean):
    tempSkew = 0
    flatChannel = channel.ravel().tolist()
    for i in flatChannel:
        tempSkew += pow(i-mean,3)
    hasilSkew = round(np.cbrt(tempSkew/len(flatChannel)),2)
    return hasilSkew

#Excel
def getSkewExcel(channel,mean,var):
    tempSkew = 0
    flatChannel = channel.ravel().tolist()
    for i in flatChannel:
        tempSkew += pow(i-mean,3)
    hasilSkew = round((tempSkew/np.cbrt(var))/len(flatChannel),2)
    return hasilSkew

arr = np.array([5,5,5,8,8,8,10,12,12,13,13,13,13,17,50,54,54,53])
arr = np.array([1,1,2,2,2,3,3,3,3,3,4,4,4,5,1200])

numpySTD = arr.std()
meanManual = getMean(arr)
STDManual = getSTD(arr,meanManual)
varManual = getVar(arr,meanManual)

skewManual = getSkewStricker(arr,meanManual)
skewExcel = getSkewExcel(arr,meanManual,varManual)

