# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:11:59 2024

@author: admin
"""

a = [1,2,3,4,5,6356,24]

print(a[-2:])

profile_list = [(i,20-i) for i in range(1,20)]
print(profile_list)

(num_builders, num_searchers) = profile_list[0]

print(num_builders,num_searchers)

import numpy as np 
C = np.zeros((2,2))
print(C[0][1])

a=np.array([0,1,2,3,4,5,6,7,8,9])
b= np.array([0,10,10,10,10,10,10,10,10,10])
print(np.prod(a[:9]/b[:9]))
print(a[1:9]/b[1:9])