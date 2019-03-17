#-*- coding: utf-8 -*-
"""
Created on Sat Jun  9 11:03:15 2018

@author: Coder-Ze
"""

import os
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator
from matplotlib.ticker import  FormatStrFormatter

path1="E:\\wz\\论文\\时域图\\"



fault = ("N","I07","I14","I21","B07","B14","B21","O07","O14","O21")

#fault_name=("Normal","Incipient inner race fault","Moderate inner race fault","Serious inner race fault",
#            "Incipient rolling element fault","Moderate rolling element fault","Serious rolling element fault",
#            "Incipient outer race fault","Moderate outer race fault","Serious outer race fault")

fault_name=("正常状态","内圈0.18mm故障","内圈0.36mm故障","内圈0.54mm故障",
            "滚动体0.18mm故障","滚动体0.36mm故障","滚动体0.54mm故障",
            "外圈0.18mm故障","外圈0.36mm故障","外圈0.54mm故障")

def Normalize(data,style):
    max=np.max(data)
    min=np.min(data)
    #选择归一化到（0,1）还是（-1,1）
    if style==(0,1):
        Normalized_data=(data-min)/(max-min)
    elif style==(-1,1):
        Normalized_data=2*(data-min)/(max-min)-1
    return Normalized_data


#import matplotlib
mpl.rcParams['xtick.direction'] = 'out'

mpl.rcParams['figure.figsize']=(10,6)
linewidth=0.7
#x_min,x_max=0,1200
#y_min,y_max=-1,1
ftsize=14
save_path="C:\\Users\\昊天维业PC\\Desktop\\时域图\\"
#plt.subplot()

left  = 0.12
right = 0.9
bottom = 0.12 
top = 0.9   
 
wspace = 0.25
hspace = 1.6

a = plt.figure()
for e,i in enumerate(fault):
    
    plt.subplot(5,2,e+1)
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    x = np.loadtxt(path1 + i +".txt")[0:1200]
    #    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax=plt.gca()
    
    plt.xticks(np.arange(0, 1201, 240),['0', '0.02', '0.04', '0.06', '0.08','0.10'],fontsize=ftsize)
#    ax.set_xticklabels(('0', '0.02', '0.04', '0.06','0.08','0.1'))
#    plt.xticks(['0', '0.02', '0.04', '0.06','0.08','0.1'])
    
    plt.plot(x,linewidth=linewidth,color="k")
    plt.yticks(fontsize=ftsize)
#    plt.xlabel(r"Time(s)",{'family' : 'Times New Roman','weight':'normal','size':10})
#    plt.ylabel(r'$\mathrm{Amplitude(m·s^-}$'+r'$\mathrm{^2})$', 
#                          {'family' : 'Times New Roman','weight':'normal','size':ftsize})
    plt.title(fault_name[e])
    plt.xlim(0,1200)
    if np.max(x) > 0.6:
        plt.ylim(-2.5, 2.5)
    else:
        plt.ylim(-0.5,0.5)

a.text(0.03, 0.6, r'$\mathrm{Amplitude(m·s^-}$'+r'$\mathrm{^2})$',
                            {'family' : 'Times New Roman','weight':'normal','size':ftsize},
                            ha='left',va='top',
                            rotation=90)

a.text(0.5, 0.05, r"Time(s)",{'family' : 'Times New Roman','weight':'normal','size':ftsize},
                            ha='left',va='top',
                            )
#plt.savefig(save_path+"10.jpg",dpi=100)

