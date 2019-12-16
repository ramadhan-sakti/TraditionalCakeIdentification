# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:25:22 2019

@author: adhan
"""

import os
i=0

jajan = 'Lumpur'

for filename in os.listdir('jajanan_kotor/'+jajan):
        print(filename)
        try:
            os.rename('jajanan_kotor/'+jajan+'/'+filename,'jajanan_kotor/'+jajan+'/'+jajan+'_'+str(i)+'.jpg')
        except:
            continue
        i+=1
print('jml=',i)

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