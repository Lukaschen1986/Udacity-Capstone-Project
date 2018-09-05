# -*- coding: utf-8 -*-
# base
# data science
import numpy as np
import pandas as pd
from scipy.stats import levene, ttest_ind, f_oneway
#from scipy.stats import ttest_rel, mannwhitneyu, skew, kurtosis
#from scipy.stats.mstats import kruskalwallis
#from statsmodels.formula.api import ols
#from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# mechine learning
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
# plot
import matplotlib.pyplot as plt
import seaborn as sns

# 基础特征工程-1
def data_transform_1(df, is_train):
    # 缺失值处理
    df.loc[df.Open.isnull(), "Open"] = 1 # Open为空的门店默认视为营业
    df.loc[df.PromoInterval.isnull(), "PromoInterval"] = "" # PromoInterval为nan，置为""
    # 枚举值替换, a = public holiday, b = Easter holiday, c = Christmas, 0 = None
    mappings = {"0":0, "a":1, "b":2, "c":3, "d":4}
    df.StoreType = df.StoreType.replace(mappings)
    df.Assortment = df.Assortment.replace(mappings)
    df.StateHoliday = df.StateHoliday.replace(mappings)
    # 构造时间特征
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    df["DayOfWeek"] = df.Date.dt.dayofweek + 1
    df["WeekOfYear"] = df.Date.dt.weekofyear
    df["DayOfYear"] = df.Date.dt.dayofyear
    df["Tenday"] = np.nan
    df.loc[df.Day <= 10, "Tenday"] = 0
    df.loc[(df.Day >= 11) & (df.Day <= 20), "Tenday"] = 1
    df.loc[df.Day >= 21, "Tenday"] = 2
    df.Tenday = df.Tenday.astype(np.int64)    
    # 获取销售日当天 对应的 竞争对手开业 累计天数，置换为1和0
    df["CompetitionOpen"] = 365*(df.Year-df.CompetitionOpenSinceYear) + 30*(df.Month-df.CompetitionOpenSinceMonth) + (df.Day-df.CompetitionOpenSinceDay)
    df.CompetitionOpen = df.CompetitionOpen.apply(lambda x: 0 if x < 0 else x)
    df.CompetitionOpen = df.CompetitionOpen.apply(lambda x: 1 if x >= 1 else 0)
    # 用是否开业修改竞争对手距离，如果竞争对手尚未开业，则距离为0
    df.loc[df.CompetitionOpen == 0, "CompetitionDistance"] = 0
    # 获取销售日当天 对应的 长促 累计月数
    df["Promo2Open"] = 12*(df.Year-df.Promo2SinceYear) + (df.WeekOfYear-df.Promo2SinceWeek)/4.0
    df.Promo2Open = df.Promo2Open.apply(lambda x: 0 if x < 0 else x)
    # 用长促是否存在和对应的月份，添加门店是否处在长促中
    MonthStr = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sept", 10:"Oct", 11:"Nov", 12:"Dec"}
    df["MonthStr"] = df.Month.map(MonthStr)
    df["InPromo2"] = 0
    for interval in df.PromoInterval.unique():
        if interval != "":
            for month in interval.split(","):
                df.loc[(df.MonthStr == month) & (df.Promo2Open > 0), "InPromo2"] = 1
    # 特征序列
    if is_train:
        features = ["Store","Date","DayOfWeek","Sales","Open","Promo","StateHoliday","SchoolHoliday",\
                    "StoreType","Assortment","CompetitionDistance","Year","Month","Day","WeekOfYear",\
                    "DayOfYear","Tenday","CompetitionOpen","InPromo2"]
    else:
        features = ["Id","Store","Date","DayOfWeek","Open","Promo","StateHoliday","SchoolHoliday",\
                    "StoreType","Assortment","CompetitionDistance","Year","Month","Day","WeekOfYear",\
                    "DayOfYear","Tenday","CompetitionOpen","InPromo2"]
    df = df[features]
    # 调整字段类型
    df.Store = df.Store.astype(np.int64)
    df.DayOfWeek = df.DayOfWeek.astype(np.int64)
    df.Open = df.Open.astype(np.int64)
    df.Promo = df.Promo.astype(np.int64)
    df.StateHoliday = df.StateHoliday.astype(np.int64)
    df.SchoolHoliday = df.SchoolHoliday.astype(np.int64)
    df.StoreType = df.StoreType.astype(np.int64)
    df.Assortment = df.Assortment.astype(np.int64)
    df.CompetitionDistance = df.CompetitionDistance.astype(np.float64)
    df.Year = df.Year.astype(np.int64)
    df.Month = df.Month.astype(np.int64)
    df.Day = df.Day.astype(np.int64)
    df.WeekOfYear = df.WeekOfYear.astype(np.float64)
    df.DayOfYear = df.DayOfYear.astype(np.float64)
    df.Tenday = df.Tenday.astype(np.int64)
    df.CompetitionOpen = df.CompetitionOpen.astype(np.int64)
    df.InPromo2 = df.InPromo2.astype(np.int64)
    return df


# 基础特征工程-2
def data_transform_2(df):
    # 店铺列表
    store_list = list(set(df.Store))
    # 新增特征
    df["WillClosedTomorrow_TodayIsSat"] = 0
    df["WillClosedTomorrow_TodayIsNotSat"] = 0
    df["WasClosedYesterday_TodayIsMon"] = 0
    df["WasClosedYesterday_TodayIsNotMon"] = 0
    # 对每个店铺做循环
    t0 = pd.Timestamp.now()
    log = []
    for store in store_list:
        # 获取单个店铺数据
        df_tmp = df[df.Store == store]
        # 计算明天时，Open数据向上移一个单位
        df_tmp["Open_m1"] = df_tmp.Open.shift(-1)
        df_tmp.loc[(df_tmp.DayOfWeek == 6) & (df_tmp.Open_m1 == 0), "WillClosedTomorrow_TodayIsSat"] = 1
        df_tmp.loc[(df_tmp.DayOfWeek != 6) & (df_tmp.Open_m1 == 0), "WillClosedTomorrow_TodayIsNotSat"] = 1
        # 获取对应的索引
        idx_1 = df_tmp[df_tmp.WillClosedTomorrow_TodayIsSat == 1].index
        idx_2 = df_tmp[df_tmp.WillClosedTomorrow_TodayIsNotSat == 1].index
        # 将索引对应的值赋值到总表
        df.loc[idx_1, "WillClosedTomorrow_TodayIsSat"] = 1
        df.loc[idx_2, "WillClosedTomorrow_TodayIsNotSat"] = 1
        # 计算昨天时，Open数据向下移一个单位
        df_tmp["Open_p1"] = df_tmp.Open.shift(1)
        df_tmp.loc[(df_tmp.DayOfWeek == 1) & (df_tmp.Open_p1 == 0), "WasClosedYesterday_TodayIsMon"] = 1
        df_tmp.loc[(df_tmp.DayOfWeek != 1) & (df_tmp.Open_p1 == 0), "WasClosedYesterday_TodayIsNotMon"] = 1
        # 获取对应的索引
        idx_3 = df_tmp[df_tmp.WasClosedYesterday_TodayIsMon == 1].index
        idx_4 = df_tmp[df_tmp.WasClosedYesterday_TodayIsNotMon == 1].index
        # 将索引对应的值赋值到总表
        df.loc[idx_3, "WasClosedYesterday_TodayIsMon"] = 1
        df.loc[idx_4, "WasClosedYesterday_TodayIsNotMon"] = 1
        # 打印计算日志
        log.append(store)
        if len(log) % 100 == 0:
            print("已计算 %d 家门店" % len(log))
    t1 = pd.Timestamp.now()
    print(t1-t0)
    # 调整字段类型
    df.WillClosedTomorrow_TodayIsSat = df.WillClosedTomorrow_TodayIsSat.astype(np.int64)
    df.WillClosedTomorrow_TodayIsNotSat = df.WillClosedTomorrow_TodayIsNotSat.astype(np.int64)
    df.WasClosedYesterday_TodayIsMon = df.WasClosedYesterday_TodayIsMon.astype(np.int64)
    df.WasClosedYesterday_TodayIsNotMon = df.WasClosedYesterday_TodayIsNotMon.astype(np.int64)
    return df

# 两独立样本T检验
def two_sample_ttest(df, x, y, val_1, val_2, W, H, plot=True):
    # mean_compare_table
    mean_compare_table = df.groupby(x, as_index=False)[[y]].mean()
    print(mean_compare_table)
    a = df.loc[df[x] == val_1, y].tolist()
    b = df.loc[df[x] == val_2, y].tolist()
    if plot:
        # plot-1
        plt.figure(figsize=(W,H))
        sns.violinplot(x, y, data=df)
        # plot-2
        plt.figure(figsize=(W,H))
        sns.kdeplot(a, shade=True, label=val_1)
        sns.kdeplot(b, shade=True, label=val_2)
    # T-test
    groups = [a, b]
    levene_test = levene(*groups)
    if levene_test.pvalue >= 0.05:
        t_test = ttest_ind(a, b, equal_var=True) # standard independent 2 sample test
    else:
        t_test = ttest_ind(a, b, equal_var=False) # Welch's t-test
    p_value = t_test.pvalue
    # 结论
    if p_value <= 0.05:
        print(p_value)
        print("%s 在 %s 上存在显著性差异" % (y, x))
    else:
        print(p_value)
        print("%s 在 %s 上不存在显著性差异" % (y, x))
    return mean_compare_table


# 单因素方差分析
def oneway_anova(df, x, y, W, H, use_hsd=True, plot=True):
    # mean_compare_table
    mean_compare_table = df.groupby(x, as_index=False)[[y]].mean()
    print(mean_compare_table)
    if plot:
        # plot
        plt.figure(figsize=(W,H))
        sns.violinplot(x, y, data=df)
    # set group
    val_list = list(set(df[x]))
    groups = []
    for val in val_list:
        groups.append(df.loc[df[x] == val, y].tolist())
    # anova
    levene_test = levene(*groups)
    if levene_test.pvalue >= 0.05:
        print("方差齐")
        f_value, p_value = f_oneway(*groups)
    else:
        print("方差不齐")
        f_value, p_value = f_oneway(*groups) # 实际都使用f_oneway
        #h_value, p_value = kruskalwallis(*groups)
    # 结论
    print(p_value)
    if use_hsd:
        hsd = pairwise_tukeyhsd(endog=df[y], groups=df[x], alpha=0.05)
        print(hsd.summary())
    return mean_compare_table


# 特征图
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    
    
# rmspe
rmspe = lambda y, y_hat: np.sqrt(np.mean((y_hat/y-1)**2))
def rmspe_xg(y_hat, y):
    y = np.expm1(y.get_label())
    y_hat = np.expm1(y_hat)
    return "rmspe", rmspe(y, y_hat)

# 单体模型训练函数
def train(x_train, y_train, x_valid, y_valid, n_estimators_0, objective, eval_metric, scoring, rmspe_xg, kfold, esr):
    # 1-设置参数初始值
    print("1-设置参数初始值")
    reg = XGBRegressor(
        # General Parameters
        booster="gbtree",
        silent=1,
        nthread=-1,
        n_jobs=-1,
        # Booster Parameters
        learning_rate=0.1,
        n_estimators=n_estimators_0,
        gamma=0,
        max_depth=7,
        min_child_weight=0.001,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0,
        reg_lambda=1,
        max_delta_step=0,
        scale_pos_weight=1,
        # Learning Task Parameters
        objective=objective,
        eval_metric=eval_metric,
        seed=0
        )
    
    # 2-训练最优弱分类器个数：n_estimators_1
    print("2-训练最优弱分类器个数：n_estimators_1")
    xgb_param = reg.get_xgb_params()
    d_train = xgb.DMatrix(x_train, y_train)
    d_valid = xgb.DMatrix(x_valid, y_valid)
    watchlist = [(d_train, "train"), (d_valid, "valid")]
    
    t_begin = pd.Timestamp.now()
    xgb_cv = xgb.cv(params=xgb_param, 
                    dtrain=d_train, 
                    num_boost_round=xgb_param["n_estimators"],
                    nfold=kfold, 
                    feval=rmspe_xg,
                    #metrics=eval_metric, 
                    early_stopping_rounds=int(xgb_param["n_estimators"]/esr),
                    verbose_eval=None)
    t1 = pd.Timestamp.now()
    n_estimators_1 = xgb_cv.shape[0]
    reg.set_params(n_estimators=n_estimators_1)
    xgb_param = reg.get_xgb_params()
    print("分类器个数：%s， 用时：%s" % (n_estimators_1, (t1-t_begin)))
    
    # 3-暴力搜索：learning_rate
    print("3-暴力搜索：learning_rate")
    param = {"learning_rate": [0.1, 0.2, 0.3]} 
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_3 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    #model_3.grid_scores_; model_3.best_score_; model_3.best_estimator_
    best_param = model_3.best_params_["learning_rate"]
    reg.set_params(learning_rate=best_param)
    xgb_param = reg.get_xgb_params()
    print("learning_rate：%s， 用时：%s" % (best_param, (t1-t0)))
    
    # 4-暴力搜索：max_depth, min_child_weight
    print("4-暴力搜索：max_depth, min_child_weight")
    param = {"max_depth": [3,5,7,9,11], "min_child_weight": [0.001, 0.01, 0.1, 1]}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_4 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param_1 = model_4.best_params_["max_depth"]
    best_param_2 = model_4.best_params_["min_child_weight"]
    print("max_depth：%s，min_child_weight：%s，用时：%s" % (best_param_1, best_param_2, (t1-t0)))
    
    # 5-精确搜索：max_depth
    print("5-精确搜索：max_depth")
    param = {"max_depth": [best_param_1-1, best_param_1, best_param_1+1]}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_5 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param_1 = model_5.best_params_["max_depth"]
    reg.set_params(max_depth=best_param_1)
    xgb_param = reg.get_xgb_params()
    print("max_depth：%s，用时：%s" % (best_param_1, (t1-t0)))
    
    # 6-暴力搜索：gamma
    print("6-暴力搜索：gamma")
    param = {"gamma": [0, 0.5, 1, 1.5, 2, 2.5]}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_6 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param = model_6.best_params_["gamma"]
    print("gamma：%s，用时：%s" % (best_param, (t1-t0)))
    
    # 7-精确搜索：gamma
    print("7-精确搜索：gamma")
    if best_param == 0:
        param = {"gamma": [0, 0.1, 0.2, 0.3, 0.4]}
    else:
        param = {"gamma": np.arange(best_param-0.2, best_param+0.3, 0.1)}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_7 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param = model_7.best_params_["gamma"]
    reg.set_params(gamma=best_param)
    xgb_param = reg.get_xgb_params()
    print("gamma：%s，用时：%s" % (best_param, (t1-t0)))
    
    # 8-调整最优弱分类器个数：n_estimators_2
    print("8-调整最优弱分类器个数：n_estimators_2")
    reg.set_params(n_estimators=n_estimators_0)
    xgb_param = reg.get_xgb_params()
    
    t0 = pd.Timestamp.now()
    xgb_cv = xgb.cv(params=xgb_param, 
                    dtrain=d_train, 
                    num_boost_round=xgb_param["n_estimators"],
                    nfold=kfold, 
                    feval=rmspe_xg,
                    #metrics=eval_metric, 
                    early_stopping_rounds=int(xgb_param["n_estimators"]/esr),
                    verbose_eval=None)
    t1 = pd.Timestamp.now()
    n_estimators_2 = xgb_cv.shape[0]
    reg.set_params(n_estimators=n_estimators_2)
    xgb_param = reg.get_xgb_params()
    print("分类器个数：%s， 用时：%s" % (n_estimators_2, (t1-t0)))
    
    # 9-暴力搜索：subsample, colsample_bytree
    print("9-暴力搜索：subsample, colsample_bytree")
    param = {"subsample": [0.6,0.7,0.8,0.9], "colsample_bytree": [0.6,0.7,0.8,0.9]}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_8 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param_1 = model_8.best_params_["subsample"]
    best_param_2 = model_8.best_params_["colsample_bytree"]
    print("subsample：%s，colsample_bytree：%s，用时：%s" % (best_param_1, best_param_2, (t1-t0)))
    
    # 10-精确搜索：subsample, colsample_bytree
    print("10-精确搜索：subsample, colsample_bytree")
    param = {"subsample": [best_param_1-0.05, best_param_1, best_param_1+0.05],
             "colsample_bytree": [best_param_2-0.05, best_param_2, best_param_2+0.05]}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_9 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param_1 = model_9.best_params_["subsample"]
    best_param_2 = model_9.best_params_["colsample_bytree"]
    reg.set_params(subsample=best_param_1, colsample_bytree=best_param_2)
    xgb_param = reg.get_xgb_params()
    print("subsample：%s，colsample_bytree：%s，用时：%s" % (best_param_1, best_param_2, (t1-t0)))
    
    # 11-暴力搜索:reg_alpha
    print("11-暴力搜索:reg_alpha")
    param = {"reg_alpha": [0, 1, 2, 3]}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_11 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param = model_11.best_params_["reg_alpha"]
    reg.set_params(reg_alpha=best_param)
    xgb_param = reg.get_xgb_params()
    print("reg_alpha：%s，用时：%s" % (best_param, (t1-t0)))
    
    # 12-精确搜索：reg_alpha
    print("12-精确搜索：reg_alpha")
    if best_param == 0:
        param = {"reg_alpha": [0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    else:
        param = {"reg_alpha": np.arange(best_param-0.5, best_param+0.5, 0.2)}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_12 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param = model_12.best_params_["reg_alpha"]
    reg.set_params(reg_alpha=best_param)
    xgb_param = reg.get_xgb_params()
    print("reg_alpha：%s，用时：%s" % (best_param, (t1-t0)))
    
    # 13-暴力搜索：reg_lambda
    print("13-暴力搜索：reg_lambda")
    param = {"reg_lambda": [1,3,5,7]}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_13 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param = model_13.best_params_["reg_lambda"]
    reg.set_params(reg_lambda=best_param)
    xgb_param = reg.get_xgb_params()
    print("reg_lambda：%s，用时：%s" % (best_param, (t1-t0)))
    
    # 14-精确搜索：reg_lambda
    print("14-精确搜索：reg_lambda")
    param = {"reg_lambda": np.arange(best_param-1, best_param+1, 0.2)}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_14 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param = model_14.best_params_["reg_lambda"]
    reg.set_params(reg_lambda=best_param)
    xgb_param = reg.get_xgb_params()
    print("reg_lambda：%s，用时：%s" % (best_param, (t1-t0)))
    
    # 15-精确搜索：max_delta_step, scale_pos_weight
    print("15-精确搜索：max_delta_step, scale_pos_weight")
    param = {"max_delta_step": [0, 1, 3, 5], 
             "scale_pos_weight": [1, 3, 5, 7]}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_12 = reg_gscv.fit(x_train, y_train)
    t1 = pd.Timestamp.now()
    best_param_1 = model_12.best_params_["max_delta_step"]
    best_param_2 = model_12.best_params_["scale_pos_weight"]
    reg.set_params(max_delta_step=best_param_1, scale_pos_weight=best_param_2)
    xgb_param = reg.get_xgb_params()
    print("max_delta_step：%s，scale_pos_weight：%s，用时：%s" % (best_param_1, best_param_2, (t1-t0)))
    
    # 16-调整最优弱分类器个数：n_estimators_3
    print("16-调整最优弱分类器个数：n_estimators_3")
    reg.set_params(n_estimators=n_estimators_0)
    xgb_param = reg.get_xgb_params()
    
    t0 = pd.Timestamp.now()
    xgb_cv = xgb.cv(params=xgb_param, 
                    dtrain=d_train, 
                    num_boost_round=xgb_param["n_estimators"],
                    nfold=kfold, 
                    feval=rmspe_xg,
                    #metrics=eval_metric, 
                    early_stopping_rounds=int(xgb_param["n_estimators"]/esr),
                    verbose_eval=None)
    t1 = pd.Timestamp.now()
    n_estimators_3 = xgb_cv.shape[0]
    reg.set_params(n_estimators=n_estimators_3)
    xgb_param = reg.get_xgb_params()
    print("分类器个数：%s， 用时：%s" % (n_estimators_3, (t1-t0)))
    
    # 17-精确搜索：learning_rate
    print("17-精确搜索：learning_rate")
    lr = xgb_param["learning_rate"]
    param = {"learning_rate": [lr-0.05, lr, lr+0.05]}
    reg_gscv = GridSearchCV(estimator=reg, param_grid=param, scoring=scoring, n_jobs=-1, iid=False, cv=kfold)
    
    t0 = pd.Timestamp.now()
    model_16 = reg_gscv.fit(x_train, y_train)
    t_1 = pd.Timestamp.now()
    best_param = model_16.best_params_["learning_rate"]
    reg.set_params(learning_rate=best_param)
    xgb_param = reg.get_xgb_params()
    print("learning_rate：%s，用时：%s" % (best_param, (t_1-t0)))
    
    # 18-终极训练
    print("18-终极训练")
    model_res = xgb.train(params=xgb_param,
                          dtrain=d_train,
                          num_boost_round=xgb_param["n_estimators"],
                          evals=watchlist,
                          feval=rmspe_xg,
                          early_stopping_rounds=int(xgb_param["n_estimators"]/esr))
    t_end = pd.Timestamp.now()
    print("参数训练完毕，总用时：%s" % (t_end-t_begin))
    return model_res, reg

