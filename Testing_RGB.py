# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 17:38:53 2019

@author: adhan
"""


# =============================================================================
# Pengujian RGB_LBP
# =============================================================================

import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
import datetime

# FeatureSet_Training = pd.read_excel("FeatureSet144_Training/FeatureSet_Training_Labeled.xlsx")
# FeatureSet_Testing = pd.read_excel("FeatureSet144_Testing/FeatureSet_Testing_Labeled.xlsx")
# classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
#                             hidden_layer_sizes=(20,2),random_state=1, max_iter = 5000,
#                             nesterovs_momentum = False, shuffle = False)

# FeatureSet_Training = pd.read_excel("FeatureSet180_Training/FeatureSet180_Training_Labeled.xlsx")
# FeatureSet_Testing = pd.read_excel("FeatureSet180_Testing/FeatureSet180_Testing_Labeled.xlsx")
# classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
#                             hidden_layer_sizes=(100,50),random_state=1, max_iter = 10000,
#                             nesterovs_momentum = False, shuffle = False)

# FeatureSet_Training = pd.read_excel("FeatureSet/FeatureSet_Training_normal_new.xlsx")
# FeatureSet_Testing = pd.read_excel("FeatureSet/FeatureSet_Testing_normal_new.xlsx")
# classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
#                             hidden_layer_sizes=(40,4),random_state=1, max_iter = 10000,
#                             nesterovs_momentum = False, shuffle = False)

FeatureSet_Training = pd.read_excel("Data/DataTraining_Random.xlsx")
FeatureSet_Testing = pd.read_excel("Data/DataTesting_Random.xlsx")
classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
                            hidden_layer_sizes=(8,9),random_state=1, max_iter = 10000,
                            nesterovs_momentum = False, shuffle = False)

# =============================================================================
# Test RGB_LBP
# =============================================================================
#Input
FiturTraining_RGB_LBP = []
TargetTraining_RGB_LBP = []
FiturTesting_RGB_LBP = []
TargetTesting_RGB_LBP = []

#Data Training
for index, row in FeatureSet_Training.iterrows():
    #Append Fitur
    FiturTraining_RGB_LBP.append([row['meanR'], row['meanG'], row['meanB'],
          row['stdR'], row['stdG'], row['stdB'],
          row['skewR'], row['skewG'], row['skewB']])
    #Append Target
    TargetTraining_RGB_LBP.append(row['Class'])

#Data Testing
for index, row in FeatureSet_Testing.iterrows():
    #Append Fitur
    FiturTesting_RGB_LBP.append([row['meanR'], row['meanG'], row['meanB'],
          row['stdR'], row['stdG'], row['stdB'],
          row['skewR'], row['skewG'], row['skewB']])
    #Append Target
    TargetTesting_RGB_LBP.append(row['Class'])
    
##Train RGB_LBP
classifier.fit(FiturTraining_RGB_LBP,TargetTraining_RGB_LBP)
##

##Predict
Result_RGB_LBP = classifier.predict(FiturTesting_RGB_LBP)
Result_RGB_LBP = Result_RGB_LBP.tolist()
##

###Akurasi
countAccu = 0
for i in range(len(Result_RGB_LBP)):
    if(Result_RGB_LBP[i] == TargetTesting_RGB_LBP[i]):
        countAccu += 1
Accu_RGB_LBP = round((countAccu/len(Result_RGB_LBP)) * 100,2)
###

# =============================================================================
# Confusion Matrix
# =============================================================================
colConfMatrix = ['Kue Ape', 'Dadar Gulung', 'Kue Lumpur', 'Putu Ayu', 'Kue Soes']
namaJajan = ['Kue Ape', 'Dadar Gulung', 'Kue Lumpur', 'Putu Ayu', 'Kue Soes']
ConfMatrix = pd.DataFrame(columns = colConfMatrix)
ConfMatrix.insert(loc = 0,column = 'Nama Jajan', value = namaJajan)
ConfMatrix = ConfMatrix.fillna(value = 0)

#Penyesuaian untuk index Confusion Matrix
for i in range(len(TargetTesting_RGB_LBP)):
    if(TargetTesting_RGB_LBP[i] == 1):
        #TargetTesting_RGB_LBP[i] = 'Kue Ape'
        TargetTesting_RGB_LBP[i] = 0
    if(Result_RGB_LBP[i] == 1):
        Result_RGB_LBP[i] = 'Kue Ape'
    if(TargetTesting_RGB_LBP[i] == 2):
        #TargetTesting_RGB_LBP[i] = 'Dadar Gulung'
        TargetTesting_RGB_LBP[i] = 1
    if(Result_RGB_LBP[i] == 2):
        Result_RGB_LBP[i] = 'Dadar Gulung'
    if(TargetTesting_RGB_LBP[i] == 3):
        #TargetTesting_RGB_LBP[i] = 'Kue Lumpur'
        TargetTesting_RGB_LBP[i] = 2
    if(Result_RGB_LBP[i] == 3):
        Result_RGB_LBP[i] = 'Kue Lumpur'
    if(TargetTesting_RGB_LBP[i] == 4):
        #TargetTesting_RGB_LBP[i] = 'Putu Ayu'
        TargetTesting_RGB_LBP[i] = 3
    if(Result_RGB_LBP[i] == 4):
        Result_RGB_LBP[i] = 'Putu Ayu'
    if(TargetTesting_RGB_LBP[i] == 5):
        #TargetTesting_RGB_LBP[i] = 'Kue Soes'
        TargetTesting_RGB_LBP[i] = 4
    if(Result_RGB_LBP[i] == 5):
        Result_RGB_LBP[i] = 'Kue Soes'

#Memasukkan data ke Confusion Matrix
for i in range(len(TargetTesting_RGB_LBP)):
    target = TargetTesting_RGB_LBP[i]
    result = Result_RGB_LBP[i]
    ConfMatrix.at[target,result] += 1
        