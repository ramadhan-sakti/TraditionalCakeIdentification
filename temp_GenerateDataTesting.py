# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:08:17 2019

@author: adhan
"""
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
import datetime

# =============================================================================
# Generate Data Testing
# =============================================================================
startTime = datetime.datetime.now()

listApe = [5,9,16,19,25,61,63,66,70,78,79,81,83,85,88,91,95,97,99,102,103,105,107,
           109,113,116,122,123,128,135,137,142,160,162,173,178]
listDadarGulung = [9,15,18,51,55,58,61,64,70,79,80,86,89,92,96,99,105,109,113,
                   116,119,120,122,127,130,132,143,146,152,155,157,159,161,
                   163,172,177]
listLumpur = [25,28,37,38,42,44,47,49,52,53,54,57,59,61,63,66,81,90,106,112,
              116,117,119,126,128,134,142,146,149,150,151,152,159,166,173,177]
listPutuAyu = [16,19,21,22,26,32,36,41,43,45,49,52,55,58,61,65,66,69,75,80,85,
               87,90,93,133,137,140,147,148,153,154,158,160,162,168,175]
listSoes = [1,4,8,11,14,27,32,37,39,43,47,52,57,85,100,102,108,110,113,115,
            117,119,122,124,125,127,128,131,134,138,141,143,154,160,166,174]
listJajan = ['Ape_', 'DadarGulung_', 'Lumpur_', 'PutuAyu_', 'Soes_']

FeatureSet_All = pd.read_excel("FeatureSet_New/FeatureSet_All_Normal.xlsx")
DataTesting = pd.DataFrame(columns = FeatureSet_All.columns)

for jajan in listJajan:
    for index, row in FeatureSet_All.iterrows():
        if(jajan == 'Ape_'):
            for i in listApe:
                #Append Fitur
                if(jajan+str(i) == FeatureSet_All.iloc[index]['Nama Item']):
                    DataTesting = DataTesting.append(FeatureSet_All.iloc[index][:],ignore_index=True)
        if(jajan == 'DadarGulung_'):
            for i in listDadarGulung:
                #Append Fitur
                if(jajan+str(i) == FeatureSet_All.iloc[index]['Nama Item']):
                    DataTesting = DataTesting.append(FeatureSet_All.iloc[index][:],ignore_index=True)
        if(jajan == 'Lumpur_'):
            for i in listLumpur:
                #Append Fitur
                if(jajan+str(i) == FeatureSet_All.iloc[index]['Nama Item']):
                    DataTesting = DataTesting.append(FeatureSet_All.iloc[index][:],ignore_index=True)
        if(jajan == 'PutuAyu_'):
            for i in listPutuAyu:
                #Append Fitur
                if(jajan+str(i) == FeatureSet_All.iloc[index]['Nama Item']):
                    DataTesting = DataTesting.append(FeatureSet_All.iloc[index][:],ignore_index=True)
        if(jajan == 'Soes_'):
            for i in listSoes:
                #Append Fitur
                if(jajan+str(i) == FeatureSet_All.iloc[index]['Nama Item']):
                    DataTesting = DataTesting.append(FeatureSet_All.iloc[index][:],ignore_index=True)
                    
                    
                    

endTime = datetime.datetime.now()
deltaTime = endTime-startTime

DataTesting.to_excel('DataTesting_Random.xlsx')