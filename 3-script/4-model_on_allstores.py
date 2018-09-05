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
# mechine learning
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn import linear_model
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
# plot
import matplotlib.pyplot as plt
# func
import func_detail as fd

# 载入数据
f = open(txt_path + "dataSet_3.txt", "rb"); dataSet_3 = pickle.load(f); f.close() # 仅开业有销量数据
f = open(txt_path + "df_test.txt", "rb"); df_test = pickle.load(f); f.close() # 测试集仅开业数据
f = open(txt_path + "df_test_open.txt", "rb"); df_test_open = pickle.load(f); f.close() # 测试集仅开业数据
f = open(txt_path + "df_test_closed.txt", "rb"); df_test_closed = pickle.load(f); f.close() # 测试集仅闭店数据

# 定义需要取log的字段
dataSet_3.WeekOfYear = np.log1p(dataSet_3.WeekOfYear)
dataSet_3.DayOfYear = np.log1p(dataSet_3.DayOfYear)
dataSet_3.CompetitionDistance = np.log1p(dataSet_3.CompetitionDistance)
df_test_open.WeekOfYear = np.log1p(df_test_open.WeekOfYear)
df_test_open.DayOfYear = np.log1p(df_test_open.DayOfYear)
df_test_open.CompetitionDistance = np.log1p(df_test_open.CompetitionDistance)

# 为df_test_open新增一列Sales用于后续存储预测值
y_hat = pd.Series(np.zeros((len(df_test_open))), name="Sales", index=df_test_open.index)
df_test_open = pd.concat((y_hat, df_test_open), axis=1)

# 划分数据集
trainSet = dataSet_3[dataSet_3.Date < "2015-06-15"] # 2013-01-01 ~ 2015-06-14
validSet = dataSet_3[dataSet_3.Date >= "2015-06-15"] # 2015-06-15 ~ 2015-07-31
testSet = df_test_open # 2015-08-01 ~ 2015-09-17

# 整体体模型所用特征与变换
colnames = ["Sales", \
            "WeekOfYear","DayOfYear","CompetitionDistance", \
            "Promo","InPromo2","CompetitionOpen", \
            "WillClosedTomorrow_TodayIsSat","WillClosedTomorrow_TodayIsNotSat", \
            "WasClosedYesterday_TodayIsMon","WasClosedYesterday_TodayIsNotMon", \
            "SchoolHoliday","StateHoliday", \
            "StoreType","Assortment","StoreSales", \
            "Year","Month","Tenday","Day","DayStr","DayOfWeek","WeekOfYearStr", \
            "DayOfYearOutlier","DayOfYearSlopeStr", \
            "CompetitionState", "CompetitionDistanceStr"]

df_train = trainSet[colnames]
df_valid = validSet[colnames]
df_test = testSet[colnames]

mapper = DataFrameMapper(
        features=[
                (["Sales"], None),
                (["WeekOfYear"], None),
                (["DayOfYear"], None),
                (["CompetitionDistance"], None),
                (["Promo"], LabelBinarizer()),
                (["InPromo2"], LabelBinarizer()),
                (["CompetitionOpen"], LabelBinarizer()),
                (["WillClosedTomorrow_TodayIsSat"], LabelBinarizer()),
                (["WillClosedTomorrow_TodayIsNotSat"], LabelBinarizer()),
                (["WasClosedYesterday_TodayIsMon"], LabelBinarizer()),
                (["WasClosedYesterday_TodayIsNotMon"], LabelBinarizer()),
                (["SchoolHoliday"], LabelBinarizer()),
                (["StateHoliday"], OneHotEncoder()),
                (["StoreType"], OneHotEncoder()),
                (["Assortment"], OneHotEncoder()),
                (["StoreSales"], OneHotEncoder()),
                (["Year"], OneHotEncoder()),
                (["Month"], OneHotEncoder()),
                (["Tenday"], OneHotEncoder()),
                (["Day"], OneHotEncoder()),
                (["DayStr"], OneHotEncoder()),
                (["DayOfWeek"], OneHotEncoder()),
                (["WeekOfYearStr"], OneHotEncoder()),
                (["DayOfYearOutlier"], OneHotEncoder()),
                (["DayOfYearSlopeStr"], OneHotEncoder()),
                (["CompetitionState"], OneHotEncoder()),
                (["CompetitionDistanceStr"], OneHotEncoder())
                ],
        default=False # None 保留; False 丢弃
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

# 基线模型
reg_lm = linear_model.LinearRegression(fit_intercept=True)
reg_lm.fit(x_train, y_train)
#reg_lm.coef_
#reg_lm.intercept_
y_hat_train = reg_lm.predict(x_train)
y_hat_train = np.expm1(y_hat_train).astype(np.int64)
fd.rmspe(np.expm1(y_train), y_hat_train) # 训练集rmspe：0.3033

y_hat_valid = reg_lm.predict(x_valid)
y_hat_valid = np.expm1(y_hat_valid).astype(np.int64)
fd.rmspe(np.expm1(y_valid), y_hat_valid) # 验证集rmspe：0.2274

# 正式模型训练、验证、测试
objective = "reg:linear"; eval_metric = "rmse"; scoring="neg_mean_squared_error"; 
n_estimators=20000; learning_rate=0.01; max_depth = 9; subsample = 0.9; colsample_bytree = 0.9

d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_valid, y_valid)
watchlist = [(d_train, "train"), (d_valid, "valid")]

# model
reg = XGBRegressor(
        booster="gbtree",
        silent=1,
        n_jobs=-1,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective=objective,
        eval_metric=eval_metric,
        seed=0
        )

xgb_param = reg.get_xgb_params()
model_allstores = xgb.train(params=xgb_param,
                            dtrain=d_train,
                            num_boost_round=xgb_param["n_estimators"],
                            evals=watchlist,
                            feval=fd.rmspe_xg,
                            early_stopping_rounds=1000)
f = open(txt_path + "model_allstores.txt", "wb"); pickle.dump(model_allstores, f); f.close()

# 模型验证
y_hat_valid = model_allstores.predict(d_valid)
y_hat_valid = np.expm1(y_hat_valid).astype(np.int64)
fd.rmspe(np.expm1(y_valid), y_hat_valid) # 验证集rmspe：0.13831

# 模型预测
d_test = xgb.DMatrix(x_test)
y_hat = model_allstores.predict(d_test)
y_hat = np.expm1(y_hat).astype(np.int64)

# 将预测值保存到测试集
df_test_open.Sales = y_hat
f = open(txt_path + "df_test_open_2.txt", "wb"); pickle.dump(df_test_open, f); f.close()

# 特征重要性计算
df_train_transform_2 = pd.get_dummies(df_train, 
                                      columns=["StateHoliday","StoreType","Assortment","StoreSales", \
                                               "Year","Month","Tenday","Day","DayStr","DayOfWeek", \
                                               "WeekOfYearStr", "DayOfYearOutlier","DayOfYearSlopeStr", \
                                               "CompetitionState", "CompetitionDistanceStr"], 
                                               drop_first=False, dummy_na=False)
feat_name = df_train_transform_2.columns[1:]

fd.create_feature_map(feat_name)
fscore = model_allstores.get_fscore(fmap="xgb.fmap")
feat_imp = pd.DataFrame({"feature": list(fscore.keys()), "fscore": list(fscore.values())})
feat_imp.fscore /= np.sum(feat_imp.fscore) # 归一化
feat_imp = feat_imp.sort_values(by="fscore", ascending=True)
#feat_imp = feat_imp[0:10] # Top10特征
#feat_imp = feat_imp.sort_values(by="fscore", ascending=True) # 特征降序
feat_pic = feat_imp.plot(kind="barh", x="feature", y="fscore", legend=False, figsize=(6, 10))
plt.title("XGBoost Feature Importance")
plt.xlabel("relative importance")
#feat_fig = feat_pic.get_figure()
#feat_fig.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)


