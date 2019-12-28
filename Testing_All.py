# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 04:30:53 2019

@author: adhan
"""

# =============================================================================
# Testing
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
# Test All
# =============================================================================
#Input
FiturTraining = []
TargetTraining = []
FiturTesting = []
TargetTesting = []
TestMode = "HSV_LBP"

if (TestMode == "HSV_LBP"):
    #Data Training
    for index, row in FeatureSet_Training.iterrows():
        #Append Fitur
        FiturTraining.append([row['meanH'], row['meanS'], row['meanV'], row['meanLBP'],
                              row['stdH'], row['stdS'], row['stdV'], row['stdLBP'],
                              row['skewH'], row['skewS'], row['skewV'], row['skewLBP']])
        #Append Target
        TargetTraining.append(row['Class'])

    #Data Testing
    for index, row in FeatureSet_Testing.iterrows():
        #Append Fitur
        FiturTesting.append([row['meanH'], row['meanS'], row['meanV'], row['meanLBP'],
                             row['stdH'], row['stdS'], row['stdV'], row['stdLBP'],
                             row['skewH'], row['skewS'], row['skewV'], row['skewLBP']])
        #Append Target
        TargetTesting.append(row['Class'])
elif(TestMode == "HSV"):
    #Data Training
    for index, row in FeatureSet_Training.iterrows():
        #Append Fitur
        FiturTraining.append([row['meanH'], row['meanS'], row['meanV'],
                                      row['stdH'], row['stdS'], row['stdV'],
                                      row['skewH'], row['skewS'], row['skewV']])
        #Append Target
        TargetTraining.append(row['Class'])
        
        #Data Testing
    for index, row in FeatureSet_Testing.iterrows():
        #Append Fitur
        FiturTesting.append([row['meanH'], row['meanS'], row['meanV'],
                                     row['stdH'], row['stdS'], row['stdV'],
                                     row['skewH'], row['skewS'], row['skewV']])
        #Append Target
        TargetTesting.append(row['Class'])
elif(TestMode == "RGB_LBP"):
    #Data Training
    for index, row in FeatureSet_Training.iterrows():
        #Append Fitur
        FiturTraining.append([row['meanR'], row['meanG'], row['meanB'], row['meanLBP'],
                              row['stdR'], row['stdG'], row['stdB'], row['stdLBP'],
                              row['skewR'], row['skewG'], row['skewB'], row['skewLBP']])
        #Append Target
        TargetTraining.append(row['Class'])

    #Data Testing
    for index, row in FeatureSet_Testing.iterrows():
        #Append Fitur
        FiturTesting.append([row['meanR'], row['meanG'], row['meanB'], row['meanLBP'],
                             row['stdR'], row['stdG'], row['stdB'], row['stdLBP'],
                             row['skewR'], row['skewG'], row['skewB'], row['skewLBP']])
        TargetTesting.append(row['Class'])
elif(TestMode == "RGB"):
    #Data Training
    for index, row in FeatureSet_Training.iterrows():
        #Append Fitur
        FiturTraining.append([row['meanR'], row['meanG'], row['meanB'],
                              row['stdR'], row['stdG'], row['stdB'],
                              row['skewR'], row['skewG'], row['skewB']])
        #Append Target
        TargetTraining.append(row['Class'])
        
        #Data Testing
    for index, row in FeatureSet_Testing.iterrows():
        #Append Fitur
        FiturTesting.append([row['meanR'], row['meanG'], row['meanB'],
                                     row['stdR'], row['stdG'], row['stdB'],
                                     row['skewR'], row['skewG'], row['skewB']])
        #Append Target
        TargetTesting.append(row['Class'])
elif(TestMode == "LAB_LBP"):
    #Data Training
    for index, row in FeatureSet_Training.iterrows():
        #Append Fitur
        FiturTraining.append([row['meanlabL'], row['meanlabA'], row['meanlabB'], row['meanLBP'],
                              row['stdlabL'], row['stdlabA'], row['stdlabB'], row['stdLBP'],
                              row['skewlabL'], row['skewlabA'], row['skewlabB'], row['skewLBP']])
        #Append Target
        TargetTraining.append(row['Class'])
        
    #Data Testing
    for index, row in FeatureSet_Testing.iterrows():
        #Append Fitur
        FiturTesting.append([row['meanlabL'], row['meanlabA'], row['meanlabB'], row['meanLBP'],
                             row['stdlabL'], row['stdlabA'], row['stdlabB'], row['stdLBP'],
                             row['skewlabL'], row['skewlabA'], row['skewlabB'], row['skewLBP']])
        TargetTesting.append(row['Class'])
elif(TestMode == "LAB"):
    #Data Training
    for index, row in FeatureSet_Training.iterrows():
        #Append Fitur
        FiturTraining.append([row['meanlabL'], row['meanlabA'], row['meanlabB'],
                              row['stdlabL'], row['stdlabA'], row['stdlabB'],
                              row['skewlabL'], row['skewlabA'], row['skewlabB']])
        #Append Target
        TargetTraining.append(row['Class'])
        
        #Data Testing
    for index, row in FeatureSet_Testing.iterrows():
        #Append Fitur
        FiturTesting.append([row['meanlabL'], row['meanlabA'], row['meanlabB'],
                                     row['stdlabL'], row['stdlabA'], row['stdlabB'],
                                     row['skewlabL'], row['skewlabB'], row['skewlabB']])
        #Append Target
        TargetTesting.append(row['Class'])
elif(TestMode == "LBP"):
    #Data Training
    for index, row in FeatureSet_Training.iterrows():
        #Append Fitur
        FiturTraining.append([row['meanLBP'],row['stdLBP'], row['skewLBP']])
        #Append Target
        TargetTraining.append(row['Class'])
        
        #Data Testing
    for index, row in FeatureSet_Testing.iterrows():
        #Append Fitur
        FiturTesting.append([row['meanLBP'],row['stdLBP'], row['skewLBP']])
        #Append Target
        TargetTesting.append(row['Class'])

classifier.fit(FiturTraining,TargetTraining)
##Predict
ResultAll = classifier.predict(FiturTesting)
ResultAll = ResultAll.tolist()
##

# =============================================================================
# Tabel Hasil
# =============================================================================
tabelHasil = pd.DataFrame(columns = ['Nama Item','Kelas Sebenarnya','Kelas Prediksi'])
for i in range(len(ResultAll)):
    if(TargetTesting[i] == 1):
        TargetTesting[i] = 'Kue Ape'
    if(ResultAll[i] == 1):
        ResultAll[i] = 'Kue Ape'
    if(TargetTesting[i] == 2):
        TargetTesting[i] = 'Dadar Gulung'
    if(ResultAll[i] == 2):
        ResultAll[i] = 'Dadar Gulung'
    if(TargetTesting[i] == 3):
        TargetTesting[i] = 'Kue Lumpur'
    if(ResultAll[i] == 3):
        ResultAll[i] = 'Kue Lumpur'
    if(TargetTesting[i] == 4):
        TargetTesting[i] = 'Putu Ayu'
    if(ResultAll[i] == 4):
        ResultAll[i] = 'Putu Ayu'
    if(TargetTesting[i] == 5):
        TargetTesting[i] = 'Kue Soes'
    if(ResultAll[i] == 5):
        ResultAll[i] = 'Kue Soes'
    tabelHasil.loc[i] = FeatureSet_Testing.iloc[i][2],TargetTesting[i],ResultAll[i]

# if(TestMode == "RGB_LBP"):
#     tabelHasil.to_excel('Hasil_Pengujian_188/Tabel_Hasil_RGB_LBP.xlsx')
# elif(TestMode == "RGB"):
#     tabelHasil.to_excel('Hasil_Pengujian_188/Tabel_Hasil_RGB.xlsx')
# elif(TestMode == "HSV_LBP"):
#     tabelHasil.to_excel('Hasil_Pengujian_188/Tabel_Hasil_HSV_LBP.xlsx')
# elif(TestMode == "HSV"):
#     tabelHasil.to_excel('Hasil_Pengujian_188/Tabel_Hasil_HSV.xlsx')
# elif(TestMode == "LAB_LBP"):
#     tabelHasil.to_excel('Hasil_Pengujian_188/Tabel_Hasil_LAB_LBP.xlsx')
# elif(TestMode == "LAB"):
#     tabelHasil.to_excel('Hasil_Pengujian_188/Tabel_Hasil_LAB.xlsx')
    
# =============================================================================
# ###Akurasi
# =============================================================================
countAccu = 0
for i in range(len(ResultAll)):
    if(ResultAll[i] == TargetTesting[i]):
        countAccu += 1
Accu = round((countAccu/len(ResultAll)) * 100,2)
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
for i in range(len(TargetTesting)):
    if(TargetTesting[i] == 'Kue Ape'):
        indexConfMatrix.append(0)
    if(TargetTesting[i] == 'Dadar Gulung'):
        indexConfMatrix.append(1)
    if(TargetTesting[i] == 'Kue Lumpur'):
        indexConfMatrix.append(2)
    if(TargetTesting[i] == 'Putu Ayu'):
        indexConfMatrix.append(3)
    if(TargetTesting[i] == 'Kue Soes'):
        indexConfMatrix.append(4)

#Memasukkan data ke Confusion Matrix
for i in range(len(TargetTesting)):
    index = indexConfMatrix[i]
    result = ResultAll[i]
    ConfMatrix.at[index,result] += 1 