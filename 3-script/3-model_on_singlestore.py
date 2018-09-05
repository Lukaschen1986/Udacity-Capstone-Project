# -*- coding: utf-8 -*-
# base
import os
os.getcwd()
os.chdir("D:/Sales_prediction/script")
txt_path = "D:/Sales_prediction/txt/"
csv_path = "D:/Sales_prediction/csv/"
import pickle
import copy
import time
# data science
import random as rd
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
# plot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as gbj
import plotly.plotly
# func
import func_detail as fd

# 载入数据
f = open(txt_path + "dataSet_3.txt", "rb"); dataSet_3 = pickle.load(f); f.close() # 仅开业有销量数据
f = open(txt_path + "df_test.txt", "rb"); df_test = pickle.load(f); f.close() # 测试集仅开业数据
f = open(txt_path + "df_test_open.txt", "rb"); df_test_open = pickle.load(f); f.close() # 测试集仅开业数据
f = open(txt_path + "df_test_closed.txt", "rb"); df_test_closed = pickle.load(f); f.close() # 测试集仅闭店数据

# 筛选门店，选出需要进行单店训练的数据
store_list = list(set(df_test_open.Store))
dataSet_3 = dataSet_3[dataSet_3.Store.isin(store_list)]

# 定义需要取log的字段
dataSet_3.WeekOfYear = np.log1p(dataSet_3.WeekOfYear)
dataSet_3.DayOfYear = np.log1p(dataSet_3.DayOfYear)
df_test_open.WeekOfYear = np.log1p(df_test_open.WeekOfYear)
df_test_open.DayOfYear = np.log1p(df_test_open.DayOfYear)

# 为df_test_open新增一列Sales用于后续存储预测值
y_hat_init = pd.Series(np.zeros((len(df_test_open))), name="Sales", index=df_test_open.index)
df_test_open = pd.concat((y_hat_init, df_test_open), axis=1)

# 记录每家店铺在验证集上的rmspe
df_rmspe_valid = pd.DataFrame({"StoreID": store_list,
                               "rmspe_valid": np.nan}, columns=["StoreID","rmspe_valid"])

# 划分数据集
trainSet = dataSet_3[dataSet_3.Date < "2015-06-15"] # 2013-01-01 ~ 2015-06-14
validSet = dataSet_3[dataSet_3.Date >= "2015-06-15"] # 2015-06-15 ~ 2015-07-31
testSet = df_test_open # 2015-08-01 ~ 2015-09-17

trainSet["idx_new"] = range(len(trainSet))
validSet["idx_new"] = range(len(validSet))
testSet["idx_new"] = range(len(testSet))

# 单体模型所用特征与变换
colnames = ["Sales", \
            "WeekOfYear","DayOfYear","Promo","InPromo2","CompetitionOpen", \
            "WillClosedTomorrow_TodayIsSat","WillClosedTomorrow_TodayIsNotSat", \
            "WasClosedYesterday_TodayIsMon","WasClosedYesterday_TodayIsNotMon", \
            "SchoolHoliday", "StateHoliday", \
            "Year","Month","Tenday","Day","DayStr","DayOfWeek","WeekOfYearStr", \
            "DayOfYearOutlier","DayOfYearSlopeStr"]

df_train = trainSet[colnames]
df_valid = validSet[colnames]
df_test = testSet[colnames]

idx_train = trainSet[["idx_new","Store"]]
idx_valid = validSet[["idx_new","Store"]]
idx_test = testSet[["idx_new","Store"]]

mapper = DataFrameMapper(
        features=[
                (["Sales"], None),
                (["WeekOfYear"], None),
                (["DayOfYear"], None),
                (["Promo"], LabelBinarizer()),
                (["InPromo2"], LabelBinarizer()),
                (["CompetitionOpen"], LabelBinarizer()),
                (["WillClosedTomorrow_TodayIsSat"], LabelBinarizer()),
                (["WillClosedTomorrow_TodayIsNotSat"], LabelBinarizer()),
                (["WasClosedYesterday_TodayIsMon"], LabelBinarizer()),
                (["WasClosedYesterday_TodayIsNotMon"], LabelBinarizer()),
                (["SchoolHoliday"], LabelBinarizer()),
                (["StateHoliday"], OneHotEncoder()),
                (["Year"], OneHotEncoder()),
                (["Month"], OneHotEncoder()),
                (["Tenday"], OneHotEncoder()),
                (["Day"], OneHotEncoder()),
                (["DayStr"], OneHotEncoder()),
                (["DayOfWeek"], OneHotEncoder()),
                (["WeekOfYearStr"], OneHotEncoder()),
                (["DayOfYearOutlier"], OneHotEncoder()),
                (["DayOfYearSlopeStr"], OneHotEncoder())
                ],
        default=False
        )

mapper_fit = mapper.fit(df_train)
df_train_transform = mapper_fit.transform(df_train)
df_valid_transform = mapper_fit.transform(df_valid)
df_test_transform = mapper_fit.transform(df_test)
    
x_train = df_train_transform[:,1:]
y_train = df_train_transform[:,0]
y_train = np.log1p(y_train)

x_valid = df_valid_transform[:,1:]
y_valid = df_valid_transform[:,0]
y_valid = np.log1p(y_valid)

x_test = df_test_transform[:,1:]

# 模型训练、验证、测试
store_num = []
for store in store_list:
    # 筛选单个店铺数据
    idx_1 = idx_train.loc[idx_train.Store == store, "idx_new"]
    idx_2 = idx_valid.loc[idx_valid.Store == store, "idx_new"]
    idx_3 = idx_test.loc[idx_test.Store == store, "idx_new"]
    
    x_train_s = x_train[idx_1]
    y_train_s = y_train[idx_1]
    
    x_valid_s = x_valid[idx_2]
    y_valid_s = y_valid[idx_2]    
    
    x_test_s = x_test[idx_3]   
    
    # train
    n_estimators_0=2000; objective="reg:linear"; eval_metric="rmse"
    scoring="neg_mean_squared_error"; kfold=5; esr=10
    model_res, reg = fd.train(x_train_s, y_train_s, x_valid_s, y_valid_s, n_estimators_0, objective, eval_metric, scoring, fd.rmspe_xg, kfold, esr)
    
    # valid
    d_valid = xgb.DMatrix(x_valid_s)
    y_hat_valid = model_res.predict(d_valid)
    y_hat_valid = np.expm1(y_hat_valid).astype(np.int64)
    rmspe_valid = fd.rmspe(np.expm1(y_valid_s), y_hat_valid)
    df_rmspe_valid.loc[df_rmspe_valid.StoreID == store, "rmspe_valid"] = rmspe_valid
    
    # test
    d_test = xgb.DMatrix(x_test_s)
    y_hat = model_res.predict(d_test)
    y_hat = np.expm1(y_hat).astype(np.int64)
    
    # 将预测值保存到测试集
    df_test_open.loc[idx_3, "Sales"] = y_hat
    f = open(txt_path + "df_test_open.txt", "wb"); pickle.dump(df_test_open, f); f.close()
    
    store_num.append(store)
    print("%d stores has finished" % len(store_num))

# 保存单体模型预测结果
f = open(txt_path + "df_test_open_1.txt", "wb"); pickle.dump(df_test_open, f); f.close()

