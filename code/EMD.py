# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:57:02 2019

@author: Administrator
"""

from pyhht.emd import EMD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhht.visualization import plot_imfs

# 读取数据
#dataset = pd.read_csv('data_day.csv')
stock_dir='../dataset/AAPL.csv'
dataset = pd.read_csv(open(stock_dir),header=0)
dataset=dataset[::-1]
for col in dataset.columns:
    dataset=dataset['Open']
    data = dataset.values
    s = data.ravel()
    #emd
    decomposer = EMD(s)               
    IMF = decomposer.decompose()
    print(IMF.shape)
    imf_data = pd.DataFrame(IMF.T)
    imf_data.to_csv('../dataset/emd/emd_AAPL_'+str(col)+'.csv')
    #绘制分解图
    plot_imfs(s,IMF)

