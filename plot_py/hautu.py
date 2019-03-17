# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 04:49:24 2019

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

path1 = "E:\\wz\\tf_all\\transfer-learning-on-fault-detection\\logs\\"

casy_direct_train_acc = path1 + "direct_train_casy\\test_acc.txt"
casy_direct_train_loss = path1 + "direct_train_casy\\train_loss.txt"
tj_direct_train_acc = path1 + "direct_train_tongji\\test_acc.txt"
tj_direct_train_loss = path1 + "direct_train_tongji\\train_loss.txt"

casy_transfer_train_acc =  path1 + "transfer_from_base_to_casy\\test_acc.txt"
casy_transfer_train_loss = path1 + "transfer_from_base_to_casy\\train_loss.txt"
tj_transfer_train_acc =  path1 + "transfer_from_base_to_tongji_diff_rpm\\train_acc.txt"
tj_transfer_train_loss = path1 + "transfer_from_base_to_tongji_diff_rpm\\train_loss.txt"


num = 3
a1 = pinghua(np.loadtxt(casy_direct_train_acc), num)
l1 = pinghua(np.loadtxt(casy_direct_train_loss), num)


a2 = pinghua(np.loadtxt(casy_transfer_train_acc), num)
l2 = pinghua(np.loadtxt(casy_transfer_train_loss), num)

plt.figure(0)
plt.plot(a1, "b", label="train from scratch")
plt.plot(a2, "r", label="transfer training")
#plt.legend(handles=[a1,a2],labels=['n','1'],loc='lower left')
plt.legend(loc='lower right')
plt.ylim(0, 1)
plt.xlabel("iterations",fontsize=ftsize)
plt.ylabel("acc",fontsize=ftsize)
plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)
plt.savefig(path1 + "acc.jpg")


plt.figure(1)
plt.plot(l1, "b", label="train from scratch")
plt.plot(l2, "r", label="transfer training")
plt.legend(loc='upper right')
plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)
#plt.ylim(0, 1)
plt.xlabel("iterations",fontsize=ftsize)
plt.ylabel("loss",fontsize=ftsize)
plt.savefig(path1 + "loss.jpg")


# ######################################
# ######################################
num = 2
a3 = pinghua(np.loadtxt(tj_direct_train_acc), num)
l3 = pinghua(np.loadtxt(tj_direct_train_loss), num)

a4 = pinghua(np.loadtxt(tj_transfer_train_acc), num)
l4 = pinghua(np.loadtxt(tj_transfer_train_loss), num) 

plt.figure(2)
plt.plot(a3, "b", label="train from scratch")
plt.plot(a4, "r", label="transfer training")
#plt.legend(handles=[a1,a2],labels=['n','1'],loc='lower left')
plt.legend(loc='lower right')
plt.ylim(0, 1)
plt.xlabel("iterations",fontsize=ftsize)
plt.ylabel("acc",fontsize=ftsize)
plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)
plt.savefig(path1 + "acc1.jpg")


plt.figure(3)
plt.plot(l3, "b", label="train from scratch")
plt.plot(l4, "r", label="transfer training")
plt.legend(loc='upper right')
plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)
#plt.ylim(0, 1)
plt.xlabel("iterations",fontsize=ftsize)
plt.ylabel("loss",fontsize=ftsize)
plt.savefig(path1 + "loss1.jpg")



        