# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 17:46:14 2019

@author: adhan
"""

# =============================================================================
# Pengujian LAB
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
#                             hidden_layer_sizes=(20,2),random_state=1, max_iter = 1000000,
#                             nesterovs_momentum = False, shuffle = False)

# FeatureSet_Training = pd.read_excel("FeatureSet180_Training/FeatureSet180_Training_Labeled.xlsx")
# FeatureSet_Testing = pd.read_excel("FeatureSet180_Testing/FeatureSet180_Testing_Labeled.xlsx")
# classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
#                             hidden_layer_sizes=(100,50),random_state=1, max_iter = 10000,
#                             nesterovs_momentum = False, shuffle = False)

# FeatureSet_Training = pd.read_excel("FeatureSet/FeatureSet_Training_Normal_New.xlsx")
# FeatureSet_Testing = pd.read_excel("FeatureSet/FeatureSet_Testing_Normal_New.xlsx")
# classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
#                             hidden_layer_sizes=(40,25),random_state=1, max_iter = 20000,
#                             nesterovs_momentum = False, shuffle = False)

FeatureSet_Training = pd.read_excel("Data/DataTraining_Random.xlsx")
FeatureSet_Testing = pd.read_excel("Data/DataTesting_Random.xlsx")
classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
                            hidden_layer_sizes=(8,9),random_state=1, max_iter = 10000,
                            nesterovs_momentum = False, shuffle = False)

# =============================================================================
# Test LAB_LBP
# =============================================================================
#Input
FiturTraining_LAB_LBP = []
TargetTraining_LAB_LBP = []
FiturTesting_LAB_LBP = []
TargetTesting_LAB_LBP = []

#Data Training
for index, row in FeatureSet_Training.iterrows():
    #Append Fitur
    FiturTraining_LAB_LBP.append([row['meanlabL'], row['meanlabA'],
                                  row['meanlabB'],
                                  row['stdlabL'], row['stdlabA'],
                                  row['stdLBP'], row['skewlabL'], row['skewlabA'], 
                                  row['skewlabB']])
    #Append Target
    TargetTraining_LAB_LBP.append(row['Class'])

#Data Testing
for index, row in FeatureSet_Testing.iterrows():
    #Append Fitur
    FiturTesting_LAB_LBP.append([row['meanlabL'], row['meanlabA'], row['meanlabB'],
                                 row['stdlabL'], row['stdlabA'], 
                                 row['stdlabB'],row['skewlabL'], 
                                 row['skewlabA'], row['skewlabB']])
    #Append Target
    TargetTesting_LAB_LBP.append(row['Class'])
    
##Train LAB_LBP
classifier.fit(FiturTraining_LAB_LBP,TargetTraining_LAB_LBP)
##

##Predict
Result_LAB_LBP = classifier.predict(FiturTesting_LAB_LBP)
Result_LAB_LBP = Result_LAB_LBP.tolist()
##

###Akurasi
countAccu = 0
for i in range(len(Result_LAB_LBP)):
    if(Result_LAB_LBP[i] == TargetTesting_LAB_LBP[i]):
        countAccu += 1
Accu_LAB_LBP = round((countAccu/len(Result_LAB_LBP)) * 100,2)
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
for i in range(len(TargetTesting_LAB_LBP)):
    if(TargetTesting_LAB_LBP[i] == 1):
        #TargetTesting_LAB_LBP[i] = 'Kue Ape'
        TargetTesting_LAB_LBP[i] = 0
    if(Result_LAB_LBP[i] == 1):
        Result_LAB_LBP[i] = 'Kue Ape'
    if(TargetTesting_LAB_LBP[i] == 2):
        #TargetTesting_LAB_LBP[i] = 'Dadar Gulung'
        TargetTesting_LAB_LBP[i] = 1
    if(Result_LAB_LBP[i] == 2):
        Result_LAB_LBP[i] = 'Dadar Gulung'
    if(TargetTesting_LAB_LBP[i] == 3):
        #TargetTesting_LAB_LBP[i] = 'Kue Lumpur'
        TargetTesting_LAB_LBP[i] = 2
    if(Result_LAB_LBP[i] == 3):
        Result_LAB_LBP[i] = 'Kue Lumpur'
    if(TargetTesting_LAB_LBP[i] == 4):
        #TargetTesting_LAB_LBP[i] = 'Putu Ayu'
        TargetTesting_LAB_LBP[i] = 3
    if(Result_LAB_LBP[i] == 4):
        Result_LAB_LBP[i] = 'Putu Ayu'
    if(TargetTesting_LAB_LBP[i] == 5):
        #TargetTesting_LAB_LBP[i] = 'Kue Soes'
        TargetTesting_LAB_LBP[i] = 4
    if(Result_LAB_LBP[i] == 5):
        Result_LAB_LBP[i] = 'Kue Soes'

#Memasukkan data ke Confusion Matrix
for i in range(len(TargetTesting_LAB_LBP)):
    target = TargetTesting_LAB_LBP[i]
    result = Result_LAB_LBP[i]
    ConfMatrix.at[target,result] += 1
        