#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:34:32 2020

@author: usuario
"""


import os
os.chdir("/Users/usuario/Desktop/DATA SCIENCE RAQUEL/GitHub/Data/")


import pandas as pd

import numpy as np

pd.options.display.max_columns=30

pd.options.display.max_rows=400


def sessions_cleaning(x):

    data= pd.read_csv(x)
    
    data["user_id"] = data.user_id.fillna(method='pad')
    
    
    data.action_type = data.action_type.fillna("missings")
    
    data.action = data.action.fillna("missings")
    
    data.action_detail = data.action_detail.fillna("missings")
    
    data.device_type = data.device_type.fillna("missings")
    
    act_freq = 100000  #Frecuencia
    act = dict(zip(*np.unique(data.action_type, return_counts=True)))
    data.action_type = data.action_type.apply(lambda x: 'OTHER' if act[x] < act_freq else x)
    
    act_freq = 250000  #Frecuencia
    act = dict(zip(*np.unique(data.device_type, return_counts=True)))
    data.device_type = data.device_type.apply(lambda x: 'OTHER' if act[x] < act_freq else x)
    
    data.rename(columns={'user_id': 'id'}, inplace=True)
    
    target = pd.read_csv("target_id_train.csv")
    
    target.drop(["Unnamed: 0"], axis=1, inplace=True)
    
    dt = pd.merge(target,data, how="left")
    
    dt = dt.drop(dt[dt['action_detail']== "respond_to_alteration_request"].index)
    
    dt = dt.drop(dt[dt['action_detail']== "create_alteration_request"].index)
    
    dt = dt.drop(dt[dt['action_detail']== "special_offer_field"].index)
    
    dt = dt.drop(dt[dt['action_detail']== "guest_receipt"].index)
    
    secs_id = dt[["id","secs_elapsed"]] #5694916
    
    dt.drop(["secs_elapsed"],axis=1,inplace=True)
    
    dt.action_type = dt.action_type.fillna("0")
    
    dt.action = dt.action.fillna("0")
    
    dt.action_detail = dt.action_detail.fillna("0")
    
    dt.device_type = dt.device_type.fillna("0")
    
    a_1 = dt[dt["country_destination"] == 1][["action_detail"]]
    
    total = dt[["country_destination","action_detail"]]
    
    dt.isnull().sum()
    
    n = total.action_detail.value_counts()
    
    a_1 = a_1.action_detail.value_counts()
    
    a_1 = a_1/n
    
    a_1 =a_1.sort_values(ascending=False)
    
    a_1 = pd.DataFrame(a_1)
    a_1["action_detail_name"] = a_1.index
    
    a_1.action_detail = a_1.action_detail.fillna(0)
    
    bins = [-0.1, 0.36, 0.45, 0.55, np.inf]
    labels = ["no_book_a_detail","bene_no_book_a_detail","bene_book_a_detail","book_a_detail"]
    a_1['agrupadas_a_detail'] = pd.cut(a_1['action_detail'], bins=bins, labels=labels)
    
    a_1.drop(["action_detail"], axis=1, inplace=True)
    
    a_1.rename(columns={'action_detail_name': 'action_detail'}, inplace=True)
    
    data = pd.merge(a_1,dt, how="left")
    
    reser = data[data["country_destination"] == 1][["action"]]
    
    n_1 = reser.action.value_counts()
    
    total = dt[["country_destination","action"]]
    
    n = total.action.value_counts()
    
    act1 = n_1/n
    
    act1 =act1.sort_values(ascending=False)
    
    act1 = act1.fillna(0)
    
    act1 = pd.DataFrame(act1)
    act1["action_name"] = act1.index
    
    act1 = act1.fillna(0)
    
    bins = [-0.1, 0.36, 0.45, 0.55, np.inf]
    labels = ["no_book_action","bene_no_book_action","bene_book_action","book_action"]
    act1['agrupadas_action'] = pd.cut(act1['action'], bins=bins, labels=labels)
    
    act1.drop(["action"], axis=1, inplace=True)
    
    act1.rename(columns={'action_name': 'action'}, inplace=True)
    
    
    data2 = pd.merge(act1,data, how="left")
    
    data2.isnull().sum()
    
    
    data2.drop(["action","action_detail"], axis=1, inplace=True)
    
    data2.action_type.value_counts()
    
    logs = secs_id.groupby("id", as_index = False) [["secs_elapsed"]].mean().round()
    
    logs.isnull().sum()
    
    logs["secs_elapsed"] = logs["secs_elapsed"].fillna(0).round()
    
    g_1 = data2.groupby('id')['agrupadas_a_detail'].value_counts().unstack().fillna(0)
    
    g_2 = data2.groupby('id')['agrupadas_action'].value_counts().unstack().fillna(0)
    
    g_3 = data2.groupby('id')['action_type'].value_counts().unstack().fillna(0)
    
    g_4 = data2.groupby('id')['device_type'].value_counts().unstack().fillna(0)
    g_1["id"] = g_1.index
    
    g_1.reset_index(drop=True, inplace=True)
    
    g_2["id"] = g_2.index
    
    g_2.reset_index(drop=True, inplace=True)
    
    g_3["id"] = g_3.index
    
    g_3.reset_index(drop=True, inplace=True)
    
    g_4["id"] = g_4.index
    
    g_4.reset_index(drop=True, inplace=True)
    
    dt = pd.merge(logs,g_1)
    
    dt = pd.merge(dt,g_2)
    
    dt = pd.merge(dt,g_3)
    
    dt = pd.merge(dt,g_4, how="left")
    
    dt = dt.fillna(0)
    
    
    dt.to_csv('Sessions_all.csv')
    
    print("Retorna dt")
    
    return dt
