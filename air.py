# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:09:49 2020

@author: khkreddy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("AirQualityUCI (1).csv",sep=';')
df.info()
df.drop(df.columns[[-1,-2]],axis=1,inplace=True)
df.isnull().any()
df.isnull().sum()
df.dropna(axis=0,inplace=True)
df.replace(',','.',regex=True,inplace=True)
df.describe()
df.drop(df.columns[[0,1]],axis=1,inplace=True)
df['CO(GT)']=df['CO(GT)'].astype('float64')
df['C6H6(GT)']=df['C6H6(GT)'].astype('float64')
df['T']=df['T'].astype('float64')
df['RH']=df['RH'].astype('float64')
df['AH']=df['AH'].astype('float64')
df.replace(-200,np.nan,regex=True,inplace=True)
df.fillna(df.mean(),inplace=True)
