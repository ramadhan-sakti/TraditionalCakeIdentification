# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:25:22 2019

@author: adhan
"""

import os
# =============================================================================
# 
# =============================================================================


# =============================================================================
# Rename Training
# =============================================================================
i = 0

for filename in os.listdir('jajanan_training/'):
        print(filename)
        try:
            os.rename('jajanan_training/'+filename,'jajanan_training/'+str(i)+'.jpg')
        except:
            continue
        i+=1
print('jml=',i)

# =============================================================================
# Rename Testing
# =============================================================================
i = 0

for filename in os.listdir('jajanan_testing/'):
        print(filename)
        try:
            os.rename('jajanan_testing/'+filename,'jajanan_testing/'+str(i)+'.jpg')
        except:
            continue
        i+=1
print('jml=',i)

# =============================================================================
# Rename Final
# =============================================================================
i = 1
jajanList = os.listdir('jajanan_segmented/Ape - Copy')
jajanList = sorted(jajanList)
jajanList = list(jajanList.sort())

for filename in jajanList:
        print(filename)
        # try:
        #     os.rename('toBeExtracted/'+filename,'toBeExtracted/'+str(i)+'.jpg')
        # except:
        #     continue
        # i+=1
print('jml=',i)

# =============================================================================
# 
# =============================================================================
