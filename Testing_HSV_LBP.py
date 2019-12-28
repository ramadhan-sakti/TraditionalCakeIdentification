# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 22:46:55 2019

@author: adhan
"""

# =============================================================================
# Pengujian HSV_LBP
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
#                             hidden_layer_sizes=(20,2),random_state=1, max_iter = 10000,
#                             nesterovs_momentum = False, shuffle = False)

# FeatureSet_Training = pd.read_excel("FeatureSet180_Training/FeatureSet180_Training_Labeled.xlsx")
# FeatureSet_Testing = pd.read_excel("FeatureSet180_Testing/FeatureSet180_Testing_Labeled.xlsx")
# classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
#                             hidden_layer_sizes=(20,2),random_state=1, max_iter = 10000,
#                             nesterovs_momentum = False, shuffle = False)

# FeatureSet_Training = pd.read_excel("FeatureSet/FeatureSet_Testing_normal_new.xlsx")
# FeatureSet_Testing = pd.read_excel("FeatureSet/FeatureSet_Testing_normal_new.xlsx")
# classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
#                             hidden_layer_sizes=(40,4),random_state=1, max_iter = 10000,
#                             nesterovs_momentum = False, shuffle = False)

# FeatureSet_Training = pd.read_excel("FeatureSet_New/FeatureSet_All_Normal.xlsx")
# FeatureSet_Testing = pd.read_excel("FeatureSet_New/FeatureSet_All_Normal.xlsx")
# classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
#                             hidden_layer_sizes=(7,4),random_state=1, max_iter = 10000,
#                             nesterovs_momentum = False, shuffle = False)

FeatureSet_Training = pd.read_excel("Data/DataTraining_Random.xlsx")
FeatureSet_Testing = pd.read_excel("Data/DataTesting_Random.xlsx")
classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
                            hidden_layer_sizes=(8,9),random_state=1, max_iter = 10000,
                            nesterovs_momentum = False, shuffle = False)

# =============================================================================
# Test HSV_LBP
# =============================================================================
#Input
FiturTraining_HSV_LBP = []
TargetTraining_HSV_LBP = []
FiturTesting_HSV_LBP = []
TargetTesting_HSV_LBP = []

#Data Training
for index, row in FeatureSet_Training.iterrows():
    #Append Fitur
    FiturTraining_HSV_LBP.append([row['meanH'], row['meanS'], row['meanV'], row['meanLBP'],
          row['stdH'], row['stdS'], row['stdV'], row['stdLBP'],
          row['skewH'], row['skewS'], row['skewV'], row['skewLBP']])
    #Append Target
    TargetTraining_HSV_LBP.append(row['Class'])

#Data Testing
for index, row in FeatureSet_Testing.iterrows():
    #Append Fitur
    FiturTesting_HSV_LBP.append([row['meanH'], row['meanS'], row['meanV'], row['meanLBP'],
          row['stdH'], row['stdS'], row['stdV'], row['stdLBP'],
          row['skewH'], row['skewS'], row['skewV'], row['skewLBP']])
    #Append Target
    TargetTesting_HSV_LBP.append(row['Class'])
    
##Train HSV_LBP
classifier.fit(FiturTraining_HSV_LBP,TargetTraining_HSV_LBP)
##

##Predict
Result_HSV_LBP = classifier.predict(FiturTesting_HSV_LBP)
Result_HSV_LBP = Result_HSV_LBP.tolist()
##

# =============================================================================
# Tabel Hasil
# =============================================================================
tabelHasil = pd.DataFrame(columns = ['Nama Item','Kelas Sebenarnya','Kelas Prediksi'])
for i in range(len(Result_HSV_LBP)):
    if(TargetTesting_HSV_LBP[i] == 1):
        TargetTesting_HSV_LBP[i] = 'Kue Ape'
    if(Result_HSV_LBP[i] == 1):
        Result_HSV_LBP[i] = 'Kue Ape'
    if(TargetTesting_HSV_LBP[i] == 2):
        TargetTesting_HSV_LBP[i] = 'Dadar Gulung'
    if(Result_HSV_LBP[i] == 2):
        Result_HSV_LBP[i] = 'Dadar Gulung'
    if(TargetTesting_HSV_LBP[i] == 3):
        TargetTesting_HSV_LBP[i] = 'Kue Lumpur'
    if(Result_HSV_LBP[i] == 3):
        Result_HSV_LBP[i] = 'Kue Lumpur'
    if(TargetTesting_HSV_LBP[i] == 4):
        TargetTesting_HSV_LBP[i] = 'Putu Ayu'
    if(Result_HSV_LBP[i] == 4):
        Result_HSV_LBP[i] = 'Putu Ayu'
    if(TargetTesting_HSV_LBP[i] == 5):
        TargetTesting_HSV_LBP[i] = 'Kue Soes'
    if(Result_HSV_LBP[i] == 5):
        Result_HSV_LBP[i] = 'Kue Soes'
    
    tabelHasil.loc[i] = FeatureSet_Testing.iloc[i][2],TargetTesting_HSV_LBP[i],Result_HSV_LBP[i]

tabelHasil.to_excel('Hasil_Pengujian_188/Tabel_Hasil_HSV_LBP.xlsx')
    

# =============================================================================
# ###Akurasi
# =============================================================================
countAccu = 0
for i in range(len(Result_HSV_LBP)):
    if(Result_HSV_LBP[i] == TargetTesting_HSV_LBP[i]):
        countAccu += 1
Accu_HSV_LBP = round((countAccu/len(Result_HSV_LBP)) * 100,2)
###

# =============================================================================
# Confusion Matrix
# =============================================================================
colConfMatrix = ['Kue Ape', 'Dadar Gulung', 'Kue Lumpur', 'Putu Ayu', 'Kue Soes']
namaJajan = ['Kue Ape', 'Dadar Gulung', 'Kue Lumpur', 'Putu Ayu', 'Kue Soes']
ConfMatrix = pd.DataFrame(columns = colConfMatrix)
ConfMatrix.insert(loc = 0,column = 'Nama Jajan', value = namaJajan)
ConfMatrix = ConfMatrix.fillna(value = 0)
indexConfMatrix = []

#Penyesuaian untuk index Confusion Matrix
for i in range(len(TargetTesting_HSV_LBP)):
    if(TargetTesting_HSV_LBP[i] == 'Kue Ape'):
        indexConfMatrix.append(0)
    if(TargetTesting_HSV_LBP[i] == 'Dadar Gulung'):
        indexConfMatrix.append(1)
    if(TargetTesting_HSV_LBP[i] == 'Kue Lumpur'):
        indexConfMatrix.append(2)
    if(TargetTesting_HSV_LBP[i] == 'Putu Ayu'):
        indexConfMatrix.append(3)
    if(TargetTesting_HSV_LBP[i] == 'Kue Soes'):
        indexConfMatrix.append(4)

#Memasukkan data ke Confusion Matrix
for i in range(len(TargetTesting_HSV_LBP)):
    index = indexConfMatrix[i]
    result = Result_HSV_LBP[i]
    ConfMatrix.at[index,result] += 1 
