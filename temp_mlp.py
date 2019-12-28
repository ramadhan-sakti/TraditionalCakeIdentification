# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:30:57 2019

@author: adhan
"""

from sklearn.neural_network import MLPClassifier

# =============================================================================
# Inisialisasi
# =============================================================================

x = [[0.,0.],   # x = Data Training
     [1.,1.]
    ]

y = ['lumpur','ape']       # y = Nilai Target

classifier = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5,
                           hidden_layer_sizes=(5,2),random_state=1, max_iter = 1000,
                           nesterovs_momentum = False, shuffle = False)

# =============================================================================
# Training
# =============================================================================
classifier.fit(x,y)

# =============================================================================
# Test
# =============================================================================
classifier.predict([[0,0], 
                    [1,1], 
                    [0,1]])