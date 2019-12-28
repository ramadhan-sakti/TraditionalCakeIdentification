# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:50:48 2019

@author: adhan
"""

# =============================================================================
# Test Keras
# =============================================================================
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
import datetime
# =============================================================================
# Prepare Data
# =============================================================================

TestMode = "HSV_LBP"


jmlClass = 5
FiturTraining = []
TargetTraining = []
FiturTesting = []
TargetTesting = []

FeatureSet_Training = pd.read_excel("Data/DataTraining_Random.xlsx", usecols = range(2,34))
FeatureSet_Testing = pd.read_excel("Data/DataTesting_Random.xlsx", usecols = range(2,34))

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

# =============================================================================
# Train Keras
# =============================================================================
# convert class vectors to binary class matrices
# Konversi vektor kelas 

for i in range(len(TargetTraining)):
    TargetTraining[i] -= 1

for i in range(len(TargetTesting)):
    TargetTesting[i] -= 1

TargetTrainingBiner = keras.utils.to_categorical(TargetTraining, jmlClass)
TargetTestingBiner = keras.utils.to_categorical(TargetTesting, jmlClass)

# Mengubah List fitur menjadi NumPy Array
FiturTraining = np.array(FiturTraining)
FiturTesting = np.array(FiturTesting)

# Membuat Model
sgd = SGD(lr=0.01, nesterov=False)
if(TestMode == "HSV_LBP" or TestMode == "RGB_LBP" or TestMode == "LAB_LBP"):
    print("Dengan LBP")
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(12,)))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(jmlClass, activation='sigmoid'))
    model.summary()
elif(TestMode == "HSV" or TestMode == "RGB" or TestMode == "LAB"):
    print("Tanpa LBP")
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(12,)))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(jmlClass, activation='sigmoid'))
    model.summary()
elif(TestMode == "LBP"):
    print("Hanya LBP")
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(12,)))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(jmlClass, activation='sigmoid'))
    model.summary()
    
model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])
model.fit(FiturTraining, TargetTrainingBiner, epochs = 100, batch_size=1)

# =============================================================================
# Test
# =============================================================================
ResultAll = model.predict(FiturTesting, verbose=2)

#Mengambil output terbesar sebagai kelas akhir
ResultAll = np.argmax(ResultAll,axis=1)
ResultAll = ResultAll.tolist()

#Memanfaatkan hasil klasifikasi yg berupa int sebagai index utk confusion matrix
rowConfMatrix = TargetTesting.copy()
# =============================================================================
# Membuat Tabel Hasil
# =============================================================================
tabelHasil = pd.DataFrame(columns = ['Nama Item','Kelas Sebenarnya','Kelas Prediksi'])
for i in range(len(ResultAll)):
    if(TargetTesting[i] == 0):
        TargetTesting[i] = 'Kue Ape'
    if(ResultAll[i] == 0):
        ResultAll[i] = 'Kue Ape'
    if(TargetTesting[i] == 1):
        TargetTesting[i] = 'Dadar Gulung'
    if(ResultAll[i] == 1):
        ResultAll[i] = 'Dadar Gulung'
    if(TargetTesting[i] == 2):
        TargetTesting[i] = 'Kue Lumpur'
    if(ResultAll[i] == 2):
        ResultAll[i] = 'Kue Lumpur'
    if(TargetTesting[i] == 3):
        TargetTesting[i] = 'Putu Ayu'
    if(ResultAll[i] == 3):
        ResultAll[i] = 'Putu Ayu'
    if(TargetTesting[i] == 4):
        TargetTesting[i] = 'Kue Soes'
    if(ResultAll[i] == 4):
        ResultAll[i] = 'Kue Soes'
    tabelHasil.loc[i] = FeatureSet_Testing.iloc[i][0],TargetTesting[i],ResultAll[i]

##Menyimpan tabel hasil ke Excel
    
if(TestMode == "RGB_LBP"):
    tabelHasil.to_excel('Hasil_Pengujian_180/Tabel_Hasil_RGB_LBP.xlsx')
elif(TestMode == "RGB"):
    tabelHasil.to_excel('Hasil_Pengujian_180/Tabel_Hasil_RGB.xlsx')
elif(TestMode == "HSV_LBP"):
    tabelHasil.to_excel('Hasil_Pengujian_180/Tabel_Hasil_HSV_LBP.xlsx')
elif(TestMode == "HSV"):
    tabelHasil.to_excel('Hasil_Pengujian_180/Tabel_Hasil_HSV.xlsx')
elif(TestMode == "LAB_LBP"):
    tabelHasil.to_excel('Hasil_Pengujian_180/Tabel_Hasil_LAB_LBP.xlsx')
elif(TestMode == "LAB"):
    tabelHasil.to_excel('Hasil_Pengujian_180/Tabel_Hasil_LAB.xlsx')
    
    
# =============================================================================
# Membuat Confusion Matrix
# =============================================================================
colConfMatrix = ['Kue Ape', 'Dadar Gulung', 'Kue Lumpur', 'Putu Ayu', 'Kue Soes']
namaJajan = ['Kue Ape', 'Dadar Gulung', 'Kue Lumpur', 'Putu Ayu', 'Kue Soes']
ConfMatrix = pd.DataFrame(columns = colConfMatrix)
ConfMatrix.insert(loc = 0,column = 'Nama Jajan', value = namaJajan)
ConfMatrix = ConfMatrix.fillna(value = 0)

#Memasukkan data ke Confusion Matrix
for i in range(len(TargetTesting)):
    rowConf = rowConfMatrix[i]
    colConf = ResultAll[i]
    ConfMatrix.at[rowConf,colConf] += 1 

print(ConfMatrix)

# =============================================================================
# Menghitung Akurasi
# =============================================================================
countAccu = 0
for i in range(len(ResultAll)):
    if(ResultAll[i] == TargetTesting[i]):
        countAccu += 1
Accu = round((countAccu/len(ResultAll)) * 100,2)
print("Akurasi = "+str(Accu)+"%")