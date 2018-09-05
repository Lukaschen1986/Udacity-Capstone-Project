# -*- coding: utf-8 -*-
# base
import os
os.chdir("D:/my_project/Python_Project/udacity/Sales_prediction-master/script")
csv_path = "D:/my_project/Python_Project/udacity/Sales_prediction-master/csv/"
txt_path = "D:/my_project/Python_Project/udacity/Sales_prediction-master/txt/"
import pickle
# data science
import numpy as np
import pandas as pd

# 载入数据
f = open(txt_path + "df_test_open_1.txt", "rb"); df_test_open_1 = pickle.load(f); f.close() # 单体模型预测结果
f = open(txt_path + "df_test_open_2.txt", "rb"); df_test_open_2 = pickle.load(f); f.close() # 整体模型预测结果
f = open(txt_path + "df_test_closed.txt", "rb"); df_test_closed = pickle.load(f); f.close() # 测试集仅闭店数据

# 合并单体模型和整体模型预测数据，加上闭店数据，加权求最终预测值
df_1 = df_test_open_1[["Id","Sales"]]
df_2 = df_test_open_2[["Id","Sales"]]
df_3 = pd.merge(df_1, df_2, on="Id", how="inner")
df_3["Sales"] = df_3.Sales_x * 0.8 + df_3.Sales_y * 0.2
df_3.Sales = df_3.Sales.astype(np.int64)
df_3 = df_3[["Id","Sales"]]

df_test_closed["Sales"] = 0
df_4 = df_test_closed[["Id","Sales"]]

df_submission = pd.concat((df_3, df_4), axis=0)
df_submission = df_submission.sort_values(by="Id", ascending=True)
df_submission.to_csv(csv_path + "df_submission.csv", index=False)
f = open(txt_path + "df_rmspe_valid.txt", "rb"); df_rmspe_valid = pickle.load(f); f.close() # 测试集仅开业数据
