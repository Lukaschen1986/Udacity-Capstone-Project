# -*- coding: utf-8 -*-
import os
os.chdir("D:/Sales_prediction/script")
txt_path = "D:/Sales_prediction/txt/"
csv_path = "D:/Sales_prediction/csv/"
import pickle
import numpy as np
import pandas as pd
import func_detail as fd

# 载入原始数据
types = {"Store": np.int64,
         "StoreType": np.str,
         "Assortment": np.str,
         "Promo2": np.int64,
         "DayOfWeek": np.int64,
         "Sales": np.int64,
         "Customers": np.int64,
         "Promo": np.int64,
         "StateHoliday": np.str,
         "SchoolHoliday": np.int64}
dataRaw = pd.read_csv(csv_path + "train.csv", dtype=types, parse_dates=["Date"], \
                      date_parser=(lambda dt: pd.to_datetime(dt, format="%Y-%m-%d")))
df_test = pd.read_csv(csv_path + "test.csv", dtype=types, parse_dates=["Date"], \
                      date_parser=(lambda dt: pd.to_datetime(dt, format="%Y-%m-%d")))
df_store = pd.read_csv(csv_path + "store.csv", dtype=types)

# 将store数据合并到训练集和测试集，形成完整数据
'''
添加一个字段，描述竞争对手开业的始发天，用于后续计算竞争对手累积营业天数
'''
df_store["CompetitionOpenSinceDay"] = np.nan
df_store.loc[df_store.CompetitionOpenSinceMonth >= 1, "CompetitionOpenSinceDay"] = 1 

dataRaw = dataRaw.sort_values(by="Date", axis=0, ascending=True) # 按时间升序排列
dataRaw = pd.merge(dataRaw, df_store, on="Store", how="left") # 合并store

df_test = df_test.sort_values(by="Date", axis=0, ascending=True) # 按时间升序排列
df_test = pd.merge(df_test, df_store, on="Store", how="left") # 合并store

# 基础特征工程-1
dataSet_1 = fd.data_transform_1(dataRaw, is_train=True)
df_test = fd.data_transform_1(df_test, is_train=False)

# 基础特征工程-2
dataSet_2 = fd.data_transform_2(dataSet_1)
df_test = fd.data_transform_2(df_test)

# 获取训练集、测试集开业数据用于后续训练与测试，提取测试集闭店数据单独保存
dataSet_3 = dataSet_2[(dataSet_2.Open != 0) & (dataSet_2.Sales > 0)]
df_test_open = df_test[df_test.Open != 0]
df_test_closed = df_test[df_test.Open == 0]

# 保存数据
f = open(txt_path + "dataSet_1.txt", "wb"); pickle.dump(dataSet_1, f); f.close() # 第一次特征工程
f = open(txt_path + "dataSet_2.txt", "wb"); pickle.dump(dataSet_2, f); f.close() # 第二次特征工程
f = open(txt_path + "dataSet_3.txt", "wb"); pickle.dump(dataSet_3, f); f.close() # 仅开业有销量数据
f = open(txt_path + "df_test.txt", "wb"); pickle.dump(df_test, f); f.close() # 测试集全部
f = open(txt_path + "df_test_open.txt", "wb"); pickle.dump(df_test_open, f); f.close() # 测试集仅开业数据
f = open(txt_path + "df_test_closed.txt", "wb"); pickle.dump(df_test_closed, f); f.close()# 测试集仅闭店数据
