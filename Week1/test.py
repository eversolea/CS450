# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:13:41 2018

@author: austi
"""
import numpy as np

a = np.array([[2,1,6],[7,5,3]])
print(a) 
c = a[0,:].argsort() 
print(c) 
d = a[:,c] 
print(d) 
