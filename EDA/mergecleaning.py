#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:27:27 2020

@author: usuario
"""



import os

import pandas as pd

import numpy as np

from train_cleaning import *

os.chdir("/Users/usuario/Desktop/DATA SCIENCE RAQUEL/GitHub/EDA/")

from sessions_cleaning import *

os.chdir("/Users/usuario/Desktop/DATA SCIENCE RAQUEL/GitHub/EDA/")

import os
os.chdir("/Users/usuario/Desktop/DATA SCIENCE RAQUEL/GitHub/Data/")

def mergetrainsessions(x,sess):
    X = train_cleaning("train_users_2.csv")
    sess = sessions_cleaning("sessions.csv")
    
    x = pd.merge(X,sess, how="left")
    
    x = x.replace(np.nan, 0) #inplace=True
    
    x.drop(["id"], axis=1,inplace=True)
    
    x.dtypes
    
    data = pd.get_dummies(x,drop_first=True)
    
    data.to_csv('train_sessions_all_clean.csv')
    
    print("Nuevo DataFrame: df ", data)
    
    return data







