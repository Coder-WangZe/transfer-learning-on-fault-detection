# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 04:24:42 2019

@author: admin
"""


import os
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator
from matplotlib.ticker import  FormatStrFormatter

path1="E:\\wz\\tf_all\\simulate_data\\"


fault = ("ball","inner","outer","normal")

def Normalize(data,style):
    max=np.max(data)
    min=np.min(data)
    #选择归一化到（0,1）还是（-1,1）
    if style==(0,1):
        Normalized_data=(data-min)/(max-min)
    elif style==(-1,1):
        Normalized_data=2*(data-min)/(max-min)-1
    return Normalized_data


mpl.rcParams['figure.figsize']=(10,6)
linewidth=0.7
#x_min,x_max=0,1200
#y_min,y_max=-1,1
ftsize=14
save_path="E:\\wz\\tf_all\\simulate_data\\fig\\"


num = 9
T = 5400
for f in fault:
    txt_path = path1 + f + ".txt"
    data = np.loadtxt(txt_path)
    data = Normalize(data, (-1,1))
    print("length of f data:", len(data))
    for i in range(num):
        y = data[i*300:i*300 + T]
        a = plt.figure()
        plt.plot(y)
#        plt.axis("off")
        plt.savefig(save_path + f + "\\" + f +"_" + str(i) + ".jpg")











