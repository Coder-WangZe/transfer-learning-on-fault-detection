# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 06:33:12 2019

@author: admin
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def pinghua(array, num=3):
    res = np.empty(shape=[array.shape[0]//3, 1])
    for i in range(res.shape[0]):
        res[i] = (array[i*3] + array[i*3+1] + array[i*3+2])/3
    return res

ftsize=16
mpl.rcParams['figure.figsize']=(10,6)

path1 = "E:\\wz\\tf_all\\transfer-learning-on-fault-detection\\logs\\tongji_diff_sample_size\\"

#sample_size = np.arange(1, 8)

#train_acc_path = path1 + "snr=" + str(snr) + "_train_acc.txt"
#train_acc_path = path1 + "snr=" + str(snr) + "_train_acc.txt"
res = []

for i in range(8):
    snr = i + 1
    file_path = path1  + str(snr) + "00test_acc.txt"
    data = float((pinghua(np.loadtxt(file_path), 1))[-10])
    res.append(data)
res.reverse()

noise = np.arange(1, 9) * 100
plt.scatter(noise, res, linewidths=8)
plt.plot(noise, res, linewidth=3)
plt.ylim(0,1.1)
plt.xlim(100,900)
plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)
plt.xticks([0,100,200,300,400,500,600,700,800])
plt.xlabel("sample size ",fontsize=ftsize)
plt.ylabel("acc",fontsize=ftsize)
