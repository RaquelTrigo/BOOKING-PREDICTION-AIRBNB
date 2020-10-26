#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:37:26 2020

@author: usuario
"""


import os
os.chdir("/Users/usuario/Desktop/DATA SCIENCE RAQUEL/GitHub/")


import pandas as pd

pd.options.display.max_columns=30

pd.options.display.max_rows=100

import numpy as np

def train_users(x):
    
    data = pd.read_csv(x)
    
    data["country_destination"] = data.country_destination.replace({"FR": "BOOKING", "IT":                                      "BOOKING", "GB": "BOOKING","ES" : "BOOKING","CA": "BOOKING", "DE" : "BOOKING", "NL" : "BOOKING", "AU" : "BOOKING", "PT" : "BOOKING", "US" : "BOOKING","other" : "BOOKING"})
    
    data["country_destination"].value_counts()
    
    data["country_destination"] = data.country_destination.replace({"NDF": 0, "BOOKING": 1})
    
    data["country_destination"].value_counts()
    
    X = data.copy()
    
    X.drop(["date_first_booking"],axis=1, inplace=True)
    
    X["age"] = np.where(X['age'] < 16, np.nan, X['age'])
    
    X["age"] = np.where(X['age'] > 95, np.nan, X['age'])
    
    X["age"] = X["age"].fillna(X["age"].mean()).round()
    
    bins = [15,20,30,40,50,60,70,np.inf]
    labels = ["16-20","20-30","30-40","40-50","50-60","60-70","70+"]
    
    X["Cluster_edad"] = pd.cut(X["age"], bins=bins, labels =labels)
    
    X.drop(["age"], axis=1, inplace=True) #ELiminamos la edad y nos quedamos con nuestro target.
    
    X['first_affiliate_tracked'] = X['first_affiliate_tracked'].fillna("missings")
    
    X["first_affiliate_tracked"] = X.first_affiliate_tracked.replace({"local ops": "mk_localops","marketing": "mk_localops"})
    
    X["gender"] = X.gender.replace({"OTHER": "-unknown-"})
    
    act_freq = 10000  #Frecuencia
    act = dict(zip(*np.unique(X.first_affiliate_tracked, return_counts=True)))
    X.first_affiliate_tracked = X.first_affiliate_tracked.apply(lambda x: 'OTHER' if act[x] < act_freq else x)
    
    act_freq = 10000  #Frecuencia
    act = dict(zip(*np.unique(X.first_browser, return_counts=True)))
    X.first_browser = X.first_browser.apply(lambda x: 'OTHER' if act[x] < act_freq else x)
    
    act_freq = 20000  #Frecuencia
    act = dict(zip(*np.unique(X.affiliate_provider, return_counts=True)))
    X.affiliate_provider = X.affiliate_provider.apply(lambda x: 'OTHER' if act[x] < act_freq else x)
    
    X["signup_app"] = X.signup_app.replace({"iOS": "AppMovil","Moweb": "AppMovil","Android": "AppMovil"})
    
    X["first_device_type"] = X.first_device_type.replace({"iPad": "Other/Unknown","Desktop (Other)": "Other/Unknown","Android Tablet": "Other/Unknown","SmartPhone (Other)": "Other/Unknown"})

    X["language"] = np.where(X['language'] != "en", "other", X['language'])
    
    X["signup_method"] = X.signup_method.replace({"google": "face_google101","facebook": "face_google101"})

    dac = np.vstack(X.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
    X['dac_year'] = dac[:,0]
    X['dac_month'] = dac[:,1]
    X = X.drop(['date_account_created'], axis=1)
    
    X['timestamp_first_active'] = pd.to_datetime(X['timestamp_first_active'], format='%Y%m%d%H%M%S')
    
    X['timestamp_first_active'] = X.timestamp_first_active.map(lambda x: x.strftime('%Y-%m-%d'))
  
    tfa = np.vstack(X.timestamp_first_active.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
    X['tfa_year'] = tfa[:,0]
    X['tfa_month'] = tfa[:,1]
    X = X.drop(['timestamp_first_active'], axis=1)
    
    X.to_csv('train_users_clean.csv')
    
    return X




