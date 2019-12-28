# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:51:59 2019

@author: adhan
"""

import pandas as pd

FeatureSet = pd.read_excel("FeatureSet_New/FeatureSet_All.xlsx")
colList = []
for i in range(2,32):
    print(FeatureSet.columns[i])
    colList.append(FeatureSet.columns[i])
#Min-Max
for i in colList:
    FeatureSet[i] = (FeatureSet[i]-FeatureSet[i].min())/(FeatureSet[i].max()-FeatureSet[i].min())

FeatureSet.to_excel('FeatureSet_New/FeatureSet_All_Normal.xlsx',index = False)