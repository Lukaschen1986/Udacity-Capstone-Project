# -*- coding: utf-8 -*-
import os
os.chdir("D:/Sales_prediction/script")
txt_path = "D:/Sales_prediction/txt/"
csv_path = "D:/Sales_prediction/csv/"
import pickle
import pandas as pd
import numpy as np
from scipy.stats import levene, ttest_ind, ttest_rel, f_oneway, mannwhitneyu, skew, kurtosis
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
# plot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as gbj
import plotly.plotly
# func
import func_detail as fd

# 载入数据
f = open(txt_path + "dataSet_3.txt", "rb"); dataSet_3 = pickle.load(f); f.close() # 仅开业有销量数据
f = open(txt_path + "df_test_open.txt", "rb"); df_test_open = pickle.load(f); f.close() # 测试集仅开业数据

# 分析Store
mean_compare_Store = fd.oneway_anova(dataSet_3, x="Store", y="Sales", W=160, H=6, use_hsd=False, plot=False)
mean_compare_Store = mean_compare_Store.sort_values(by="Sales", ascending=False)
'''
思考新特征：按每家店平均销量分位点离散化
'''
q1, q2, q3, q4, q5 = np.percentile(mean_compare_Store.Sales, q=[10,25,50,75,90])
mean_compare_Store["StoreSales"] = np.nan
mean_compare_Store.loc[mean_compare_Store.Sales <= q1, "StoreSales"] = 0
mean_compare_Store.loc[(mean_compare_Store.Sales > q1) & (mean_compare_Store.Sales <= q2), "StoreSales"] = 1
mean_compare_Store.loc[(mean_compare_Store.Sales > q2) & (mean_compare_Store.Sales <= q3), "StoreSales"] = 2
mean_compare_Store.loc[(mean_compare_Store.Sales > q3) & (mean_compare_Store.Sales <= q4), "StoreSales"] = 3
mean_compare_Store.loc[(mean_compare_Store.Sales > q4) & (mean_compare_Store.Sales <= q5), "StoreSales"] = 4
mean_compare_Store.loc[mean_compare_Store.Sales > q5, "StoreSales"] = 5
# 添加新特征
dataSet_3 = pd.merge(dataSet_3, mean_compare_Store[["Store","StoreSales"]], on="Store", how="left")
dataSet_3.StoreSales = dataSet_3.StoreSales.astype(np.int64)
df_test_open = pd.merge(df_test_open, mean_compare_Store[["Store","StoreSales"]], on="Store", how="left")
df_test_open.StoreSales = df_test_open.StoreSales.astype(np.int64)
# 验证
mean_compare_Store_2 = fd.oneway_anova(dataSet_3, x="StoreSales", y="Sales", W=10, H=6, use_hsd=True, plot=False)
'''
   StoreSales         Sales
0           0   3905.316269
1           1   4942.847116
2           2   5947.667403
3           3   7225.632618
4           4   8711.041965
5           5  12048.584421

Multiple Comparison of Means - Tukey HSD,FWER=0.05
==================================================
group1 group2  meandiff   lower     upper   reject
--------------------------------------------------
  0      1    1037.5308 1010.0835 1064.9782  True 
  0      2    2042.3511 2017.1997 2067.5026  True 
  0      3    3320.3163 3295.2222 3345.4105  True 
  0      4    4805.7257 4778.3221 4833.1293  True 
  0      5    8143.2682 8113.4578 8173.0785  True 
  1      2    1004.8203  982.7642 1026.8764  True 
  1      3    2282.7855 2260.7947 2304.7763  True 
  1      4    3768.1948 3743.6015 3792.7882  True 
  1      5    7105.7373 7078.4878 7132.9868  True 
  2      3    1277.9652  1258.917 1297.0134  True 
  2      4    2763.3746 2741.3729 2785.3762  True 
  2      5     6100.917 6075.9816 6125.8524  True 
  3      4    1485.4093 1463.4732 1507.3455  True 
  3      5    4822.9518 4798.0742 4847.8294  True 
  4      5    3337.5425  3310.337 3364.7479  True 
--------------------------------------------------
'''
'''
结论：通过计算各门店销量的历史均值，将门店离散为6种类别，使得各类门店间呈现出较大的差异
新增特征：StoreSales
'''

# 分析Promo
mean_compare_Promo = fd.two_sample_ttest(dataSet_3, x="Promo", y="Sales", val_1=0, val_2=1, W=10, H=8, plot=False)
'''
   Promo        Sales
0      0  5929.826183
1      1  8228.739731
'''
'''
结论：从整体数据来看，有无短期促销对店铺的销量有显著性影响，分析发现没有短期促销时的平均销量为5929，而有短期促销平均销量是8228
'''

# 分析StateHoliday
mean_compare_StateHoliday = fd.oneway_anova(dataSet_3, x="StateHoliday", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   StateHoliday        Sales
0             0  6953.960229
1             1  8487.471182
2             2  9887.889655
3             3  9743.746479

Multiple Comparison of Means - Tukey HSD,FWER=0.05
==================================================
group1 group2  meandiff   lower     upper   reject
--------------------------------------------------
  0      1     1533.511 1230.7697 1836.2522  True 
  0      2    2933.9294 2271.8257 3596.0332  True 
  0      3    2789.7862 1843.6323 3735.9402  True 
  1      2    1400.4185  672.4879  2128.349  True 
  1      3    1256.2753  262.9431 2249.6075  True 
  2      3    -144.1432 -1298.889 1010.6027 False 
--------------------------------------------------
'''
'''
结论：州假对销量存在显著的影响，分析发现，没有州假（0）时的平均销量为6953，PublicHoliday（1）的平均销量是8487，EasterHoliday（2）的平均销量是9887，Christmas（3）的平均销量是9743。通过方差分析进一步发现除了EasterHoliday和Christmas之间差异不明显以外，其余组间均拒绝原假设，存在显著性差异（pvalue <0.05）
'''

# 分析SchoolHoliday
mean_compare_SchoolHoliday = fd.two_sample_ttest(dataSet_3, x="SchoolHoliday", y="Sales", val_1=0, val_2=1, W=10, H=8, plot=False)
'''
   SchoolHoliday        Sales
0              0  6897.207830
1              1  7200.710282
'''
'''
结论：从整体数据来看，学校假期对店铺的销量有一定影响，但并不明显，学校不放假（0）时平均销量为6897，放假（1）时为7200
'''

# 分析StoreType
mean_compare_StoreType = fd.oneway_anova(dataSet_3, x="StoreType", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   StoreType         Sales
0          1   6925.697986
1          2  10233.380141
2          3   6933.126425
3          4   6822.300064

  Multiple Comparison of Means - Tukey HSD,FWER=0.05 
=====================================================
group1 group2  meandiff    lower      upper    reject
-----------------------------------------------------
  1      2    3307.6822  3243.3717  3371.9926   True 
  1      3      7.4284    -18.7837   33.6406   False 
  1      4    -103.3979   -122.806   -83.9899   True 
  2      3    -3300.2537 -3367.7117 -3232.7958  True 
  2      4    -3411.0801 -3476.1967 -3345.9635  True 
  3      4    -110.8264  -138.9584   -82.6943   True 
-----------------------------------------------------
'''
'''
结论：从整体数据来看，店铺类型不同，销量也体现出相应的差异，其中店铺类型2的平均销量最高（10233），店铺类型4最低（6822），店铺类型1是6925，店铺类型3是6933，1和3的销量差异不够显著，其余均呈现显著性差异
'''

# 分析Assortment
dataSet_3.Assortment.value_counts()
mean_compare_Assortment = fd.oneway_anova(dataSet_3, x="Assortment", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   Assortment        Sales
0           1  6621.523057
1           2  8642.503594
2           3  7300.843547

 Multiple Comparison of Means - Tukey HSD,FWER=0.05 
====================================================
group1 group2  meandiff   lower      upper    reject
----------------------------------------------------
  1      2    2020.9805 1940.5525  2101.4086   True 
  1      3     679.3205  663.4945   695.1465   True 
  2      3     -1341.66 -1422.1879 -1261.1322  True 
----------------------------------------------------
'''
'''
结论：从整体看，店铺种类不同，销量也体现出相应的差异，其中店铺种类2的平均销量最高（8642），店铺种类1最低（6621），店铺种类3为7300
'''

# 分析CompetitionOpen
mean_compare_CompetitionOpen = pd.pivot_table(dataSet_3, index="Store", columns="CompetitionOpen", values="Sales", aggfunc=np.mean, fill_value=np.nan)
mean_compare_CompetitionOpen.columns = ["no","yes"]

sales_without_competition = mean_compare_CompetitionOpen.loc[mean_compare_CompetitionOpen.yes.isnull(), "no"] # 有357家店从头至尾都没有竞争对手
sales_always_competition = mean_compare_CompetitionOpen.loc[mean_compare_CompetitionOpen.no.isnull(), "yes"] # 有570家店从头至尾都有竞争对手
sales_pair_competition = mean_compare_CompetitionOpen.dropna(how="any", axis=0) # 有188家店此前没有竞争对手，而后存在竞争对手

# 两独立样本 t-test
groups = [sales_without_competition, sales_always_competition]
levene_test = levene(*groups)
levene_test.pvalue # 0.26532204200988235
t_test = ttest_ind(sales_without_competition, sales_always_competition, equal_var=True)
p_value = t_test.pvalue # 0.6312748113980786
plt.figure(figsize=(10,8))
sns.kdeplot(sales_without_competition, shade=True, label="without")
sns.kdeplot(sales_always_competition, shade=True, label="always")
# mean
sales_without_competition.mean() # 6902.352293987692
sales_always_competition.mean() # 6827.417098064886
# 偏度：负（左偏），正（右偏）
skew(sales_without_competition) # 2.021844978451924
skew(sales_always_competition) # 1.4583758955404342
# 峰度：负（低峰），正（尖峰）
kurtosis(sales_without_competition) # 7.572782675085753
kurtosis(sales_always_competition) # 4.599181900707614

# 配对样本 t-test
t_test = ttest_rel(sales_pair_competition.no, sales_pair_competition.yes)
p_value = t_test.pvalue # 0.0004465345309962874
plt.figure(figsize=(10,8))
sns.kdeplot(sales_pair_competition.no, shade=True, label="no")
sns.kdeplot(sales_pair_competition.yes, shade=True, label="yes")
'''
no     7409.848526
yes    7185.129756
dtype: float64
'''
idx_0 = sales_without_competition.index
idx_1 = sales_always_competition.index
idx_2 = sales_pair_competition.index
# 添加新特征
dataSet_3["CompetitionState"] = np.nan
dataSet_3.loc[dataSet_3.Store.isin(idx_0), "CompetitionState"] = 0
dataSet_3.loc[dataSet_3.Store.isin(idx_1), "CompetitionState"] = 1
dataSet_3.loc[dataSet_3.Store.isin(idx_2), "CompetitionState"] = 2
dataSet_3.CompetitionState.value_counts()
dataSet_3.CompetitionState = dataSet_3.CompetitionState.astype(np.int64)

df_test_open["CompetitionState"] = np.nan
df_test_open.loc[df_test_open.Store.isin(idx_0), "CompetitionState"] = 0
df_test_open.loc[df_test_open.Store.isin(idx_1), "CompetitionState"] = 1
df_test_open.loc[df_test_open.Store.isin(idx_2), "CompetitionState"] = 2
df_test_open.CompetitionState.value_counts()
df_test_open.CompetitionState = df_test_open.CompetitionState.astype(np.int64)

# 验证
mean_compare_CompetitionState = fd.oneway_anova(dataSet_3, x="CompetitionState", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   CompetitionState        Sales
0               0.0  6928.581664
1               1.0  6839.689367
2               2.0  7359.897653

Multiple Comparison of Means - Tukey HSD,FWER=0.05
================================================
group1 group2 meandiff   lower    upper   reject
------------------------------------------------
 0.0    1.0   -88.8923 -106.6954 -71.0892  True 
 0.0    2.0   431.316   407.5496 455.0823  True 
 1.0    2.0   520.2083  498.0143 542.4023  True 
------------------------------------------------
'''
'''
结论：
1、对竞争对手的开业状况进行分析发现，有357家店自始至终都没有竞争对手，而有570家店自始至终都存在竞争对手，另有188家店原先没有竞争对手，而后竞争对手进入商圈；
2、竞争对手的存在对销量有无影响？
2.1、对357家店和570家店进行两独立样本t-test发现，自始至终没有竞争对手和自始至终存在竞争对手的门店间销量没有显著性差别（pvalue: 0.6312），前者平均销量6902，后者6827，两类格局的门店销量数据都存在一定的右偏和尖峰，且无竞争对手的门店右偏更严重、尖峰更严重，说明无竞争对手门店的销量存在更大的极值；
2.2、对188家门店进行配对样本t-test发现，竞争对手进入的前后销量存在显著性差异（pvalue: 0.00044653）,竞争对手进入前平均销量7409，进入后平均7185，出现一定的下滑
3、加入新特征：CompetitionState
自始至终都没有竞争对手的门店：0
自始至终都有竞争对手的门店：1
此前没有后来引入竞争对手的门店：2
'''

# 分析CompetitionDistance
t_Sales_Dist = dataSet_3.groupby("Store", as_index=False)[["Sales","CompetitionDistance"]].mean()
t_Sales_Dist = t_Sales_Dist[t_Sales_Dist.CompetitionDistance != 0]
np.corrcoef(t_Sales_Dist.Sales, t_Sales_Dist.CompetitionDistance) # -0.05551006
plt.figure(figsize=(12,8))
sns.kdeplot(t_Sales_Dist.CompetitionDistance, shade=True)
# 用GMM算法筛选聚类个数
x_log = np.log1p(t_Sales_Dist.CompetitionDistance)
x_log = x_log[:, np.newaxis]
for k in np.arange(2, 11):
    gmm = GaussianMixture(n_components=k, init_params="kmeans", tol=0.001, max_iter=5000, random_state=0)
    gmm.fit(x_log)
    gmm_labels = gmm.predict(x_log)
    print(k, silhouette_score(X=x_log, labels=gmm_labels))
'''
2 0.5682768943756821
3 0.5470352836673131
4 0.5362414052882271
5 0.5278943776805186
6 0.5391203590350272
7 0.5289858777323508
8 0.5224358990040647
9 0.5305291535719682
10 0.5082490513875614
'''
gmm = GaussianMixture(n_components=2, init_params="kmeans", tol=0.001, max_iter=5000, random_state=0)
gmm.fit(x_log)
gmm_labels = gmm.predict(x_log)
t_Sales_Dist["gmm_labels"] = gmm_labels
t_Sales_Dist.groupby("gmm_labels", as_index=False)[["CompetitionDistance"]].median()
'''
   gmm_labels  CompetitionDistance
0           0           340.000000
1           1          3890.160462
'''
a = t_Sales_Dist.loc[t_Sales_Dist.gmm_labels == 0, "CompetitionDistance"]
b = t_Sales_Dist.loc[t_Sales_Dist.gmm_labels == 1, "CompetitionDistance"]
plt.figure(figsize=(12,8))
sns.kdeplot(a, shade=True, label=0)
sns.kdeplot(b, shade=True, label=1)
# 添加新特征
dataSet_3["CompetitionDistanceStr"] = 0
dataSet_3.loc[(dataSet_3.CompetitionDistance > 0) & (dataSet_3.CompetitionDistance <= 350), "CompetitionDistanceStr"] = 1
dataSet_3.loc[(dataSet_3.CompetitionDistance > 350) & (dataSet_3.CompetitionDistance <= 4000), "CompetitionDistanceStr"] = 2
dataSet_3.loc[dataSet_3.CompetitionDistance > 4000, "CompetitionDistanceStr"] = 3
dataSet_3.CompetitionDistanceStr.value_counts()

df_test_open["CompetitionDistanceStr"] = 0
df_test_open.loc[(df_test_open.CompetitionDistance > 0) & (df_test_open.CompetitionDistance <= 350), "CompetitionDistanceStr"] = 1
df_test_open.loc[(df_test_open.CompetitionDistance > 350) & (df_test_open.CompetitionDistance <= 4000), "CompetitionDistanceStr"] = 2
df_test_open.loc[df_test_open.CompetitionDistance > 4000, "CompetitionDistanceStr"] = 3
df_test_open.CompetitionDistanceStr.value_counts()
# 验证
mean_compare_CompetitionDistanceStr = fd.oneway_anova(dataSet_3, x="CompetitionDistanceStr", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   CompetitionDistanceStr        Sales
0                       0  6991.936832
1                       1  7631.197221
2                       2  6790.455592
3                       3  6849.104777
'''
'''
结论：通过计算每个门店的销量均值和竞争对手距离之间的关系，发现二者不存在显著的线性相关（pearson相关系数为-0.055），因此考虑将其离散化，通过GMM算法对竞争对手距离进行数据探索，获得聚类数为2时轮廓系数达到最大（0.5682），根据聚类中心确定类别边界为350米和4000米，创建新特征CompetitionDistanceStr，即小于350米，350到4000米，4000米以上
新特征：CompetitionDistanceStr
'''

# 分析DayOfWeek
mean_compare_DayOfWeek = fd.oneway_anova(dataSet_3, x="DayOfWeek", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   DayOfWeek        Sales
0          1  8216.252259
1          2  7088.409086
2          3  6728.786679
3          4  6768.214973
4          5  7073.034133
5          6  5875.084935
6          7  8224.723908

  Multiple Comparison of Means - Tukey HSD,FWER=0.05 
=====================================================
group1 group2  meandiff    lower      upper    reject
-----------------------------------------------------
  1      2    -1127.8432 -1161.4678 -1094.2186  True 
  1      3    -1487.4656 -1521.2077 -1453.7235  True 
  1      4    -1448.0373 -1482.2264 -1413.8481  True 
  1      5    -1143.2181 -1177.1566 -1109.2796  True 
  1      6    -2341.1673 -2374.7864 -2307.5482  True 
  1      7      8.4716   -142.2356   159.1789  False 
  2      3    -359.6224  -392.9816  -326.2633   True 
  2      4    -320.1941  -354.0054  -286.3828   True 
  2      5     -15.375    -48.9328   18.1829   False 
  2      6    -1213.3242 -1246.5589 -1180.0894  True 
  2      7    1136.3148   985.6929  1286.9368   True 
  3      4     39.4283     5.5002    73.3564    True 
  3      5     344.2475   310.5719   377.923    True 
  3      6    -853.7017  -887.0553  -820.3482   True 
  3      7    1495.9372   1345.289  1646.5855   True 
  4      5     304.8192   270.6957   338.9426   True 
  4      6     -893.13   -926.9358  -859.3243   True 
  4      7    1456.5089  1305.7599  1607.2579   True 
  5      6    -1197.9492 -1231.5015 -1164.3969  True 
  5      7    1151.6898  1000.9974  1302.3821   True 
  6      7     2349.639  2199.0182  2500.2597   True 
-----------------------------------------------------
'''
a = gbj.Scatter(x=mean_compare_DayOfWeek.DayOfWeek, y=mean_compare_DayOfWeek.Sales, opacity=1)
data = [a]
layout = dict(title="DayOfWeek")
fig = dict(data=data, layout=layout)
plotly.offline.plot(fig)

'''
结论：从整体数据来看，周因素对销量存在显著性影响，但是要注意，周一和周日、周二和周五的差异性不明显，未能拒绝原假设。
'''

# 分析Year
mean_compare_Year = fd.oneway_anova(dataSet_3, x="Year", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   Year        Sales
0  2013  6814.775168
1  2014  7026.128505
2  2015  7088.235123

Multiple Comparison of Means - Tukey HSD,FWER=0.05
===============================================
group1 group2 meandiff  lower    upper   reject
-----------------------------------------------
 2013   2014  211.3533 193.2808 229.4259  True 
 2013   2015   273.46  252.8219 294.098   True 
 2014   2015  62.1066  41.1351  83.0781   True 
-----------------------------------------------
'''
'''
结论：从整体上看，不同年份销量存在微弱差异，整体呈上升趋势，其中2013年日均销量6814，2014年日均7026，2015年日均7088
'''

# 分析Month
mean_compare_Month = fd.oneway_anova(dataSet_3, x="Month", y="Sales", W=12, H=8, use_hsd=False, plot=False)
'''
    Month        Sales
0       1  6564.298651
1       2  6589.494012
2       3  6976.817749
3       4  7046.659509
4       5  7106.808038
5       6  7001.402127
6       7  6953.575827
7       8  6649.229053
8       9  6547.469104
9      10  6602.967255
10     11  7188.554250
11     12  8608.956661
'''
a = gbj.Scatter(x=mean_compare_Month.Month, y=mean_compare_Month.Sales, opacity=1)
data = [a]
layout = dict(title="Month")
fig = dict(data=data, layout=layout)
plotly.offline.plot(fig)
'''
结论：从整体上看，不同月份销量存在显著差异，从2月份到5月份销量呈现增长趋势，6-9月出现下滑，10-12月开始飙升。
'''

# 分析Day
mean_compare_Day = fd.oneway_anova(dataSet_3, x="Day", y="Sales", W=20, H=8, use_hsd=False, plot=False)
'''
    Day        Sales
0     1  8054.505835
1     2  7987.998803
2     3  7765.916826
3     4  7746.632622
4     5  7556.054806
5     6  7149.914351
6     7  7101.614663
7     8  6785.606424
8     9  6499.517013
9    10  6429.867986
10   11  6088.286098
11   12  6186.692977
12   13  6570.339941
13   14  6606.648700
14   15  7018.797807
15   16  7314.330149
16   17  7284.416418
17   18  7340.772490
18   19  7115.279322
19   20  6955.004553
20   21  6693.696159
21   22  6544.923929
22   23  6498.481514
23   24  5916.886849
24   25  5968.280641
25   26  6190.007567
26   27  6636.996208
27   28  6943.514789
28   29  7514.074032
29   30  8355.098655
30   31  7577.710796
'''
a = gbj.Scatter(x=mean_compare_Day.Day, y=mean_compare_Day.Sales, opacity=1)
data = [a]
layout = dict(title="Day")
fig = dict(data=data, layout=layout)
plotly.offline.plot(fig)
# 添加新特征
dataSet_3["DayStr"] = 0
dataSet_3.loc[dataSet_3.Day.isin([1,2,3,16,17,18,30]), "DayStr"] = 1
dataSet_3.loc[dataSet_3.Day.isin([11,12,24,25,26]), "DayStr"] = 2
dataSet_3.DayStr.value_counts()

df_test_open["DayStr"] = 0
df_test_open.loc[df_test_open.Day.isin([1,2,3,16,17,18,30]), "DayStr"] = 1
df_test_open.loc[df_test_open.Day.isin([11,12,24,25,26]), "DayStr"] = 2
df_test_open.DayStr.value_counts()
# 验证
mean_compare_Day_2 = fd.oneway_anova(dataSet_3, x="DayStr", y="Sales", W=20, H=8, use_hsd=True, plot=False)
'''
   DayStr        Sales
0       0  6925.792458
1       1  7704.187890
2       2  6069.380683

  Multiple Comparison of Means - Tukey HSD,FWER=0.05 
=====================================================
group1 group2  meandiff    lower      upper    reject
-----------------------------------------------------
  0      1     778.3954   759.0092   797.7816   True 
  0      2    -856.4118  -878.0518  -834.7718   True 
  1      2    -1634.8072 -1660.2099 -1609.4045  True 
-----------------------------------------------------
'''
'''
结论：从整体上看，月中不同的天销量存在显著差异，每月11、12、24、25、26号普遍是销量低谷，从1号到11销量逐渐下滑，11号到18号开始上升，18号到25号又出现下滑，25号到月底又出现上升。整体呈现“W”型变化。
新特征：DayStr
'''

# 分析WeekOfYear
mean_compare_WeekOfYear = fd.oneway_anova(dataSet_3, x="WeekOfYear", y="Sales", W=30, H=8, use_hsd=False, plot=False)
'''
    WeekOfYear         Sales
0          1.0   6227.377614
1          2.0   7650.125514
2          3.0   5972.461976
3          4.0   6536.683796
4          5.0   6376.769227
5          6.0   7589.778830
6          7.0   5692.554968
7          8.0   7238.187304
8          9.0   5845.255711
9         10.0   7680.063837
10        11.0   5633.235520
11        12.0   7339.701861
12        13.0   6724.352013
13        14.0   7940.078745
14        15.0   6424.874711
15        16.0   7303.083193
16        17.0   6064.670518
17        18.0   8661.948493
18        19.0   7475.495038
19        20.0   6503.589378
20        21.0   6921.209278
21        22.0   7220.327896
22        23.0   7940.694999
23        24.0   5987.047669
24        25.0   7669.226901
25        26.0   5788.437745
26        27.0   8087.774227
27        28.0   5943.296822
28        29.0   7650.502551
29        30.0   5638.504625
30        31.0   8217.650816
31        32.0   6563.232098
32        33.0   6754.930540
33        34.0   6317.321741
34        35.0   6840.128440
35        36.0   6574.913867
36        37.0   6751.013720
37        38.0   6133.769006
38        39.0   6276.009389
39        40.0   7377.561922
40        41.0   7525.025485
41        42.0   5515.545102
42        43.0   6884.212604
43        44.0   6280.878102
44        45.0   7878.026188
45        46.0   6454.084408
46        47.0   6829.011082
47        48.0   7661.674882
48        49.0   9133.437976
49        50.0   6857.278683
50        51.0  10939.013841
51        52.0   8132.852867
'''
a = gbj.Scatter(x=mean_compare_WeekOfYear.WeekOfYear, y=mean_compare_WeekOfYear.Sales, opacity=1)
data = [a]
layout = dict(title="WeekOfYear")
fig = dict(data=data, layout=layout)
plotly.offline.plot(fig)
# 添加新特征
dataSet_3["WeekOfYearStr"] = np.nan
dataSet_3.loc[(dataSet_3.WeekOfYear <= 18) & (dataSet_3.WeekOfYear % 2 == 0), "WeekOfYearStr"] = 0
dataSet_3.loc[(dataSet_3.WeekOfYear <= 18) & (dataSet_3.WeekOfYear % 2 == 1), "WeekOfYearStr"] = 1
dataSet_3.loc[(dataSet_3.WeekOfYear >= 19) & (dataSet_3.WeekOfYear % 2 == 0), "WeekOfYearStr"] = 2
dataSet_3.loc[(dataSet_3.WeekOfYear >= 19) & (dataSet_3.WeekOfYear % 2 == 1), "WeekOfYearStr"] = 3
dataSet_3.WeekOfYearStr.value_counts()
dataSet_3.WeekOfYearStr = dataSet_3.WeekOfYearStr.astype(np.int64)

df_test_open.loc[(df_test_open.WeekOfYear <= 18) & (df_test_open.WeekOfYear % 2 == 0), "WeekOfYearStr"] = 0
df_test_open.loc[(df_test_open.WeekOfYear <= 18) & (df_test_open.WeekOfYear % 2 == 1), "WeekOfYearStr"] = 1
df_test_open.loc[(df_test_open.WeekOfYear >= 19) & (df_test_open.WeekOfYear % 2 == 0), "WeekOfYearStr"] = 2
df_test_open.loc[(df_test_open.WeekOfYear >= 19) & (df_test_open.WeekOfYear % 2 == 1), "WeekOfYearStr"] = 3
df_test_open.WeekOfYearStr = df_test_open.WeekOfYearStr.astype(np.int64)

mean_compare_WeekOfYearStr = fd.oneway_anova(dataSet_3, x="WeekOfYearStr", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   WeekOfYearStr        Sales
0            0.0  7524.668673
1            1.0  6097.971581
2            2.0  6428.309740
3            3.0  7646.824201

  Multiple Comparison of Means - Tukey HSD,FWER=0.05 
=====================================================
group1 group2  meandiff    lower      upper    reject
-----------------------------------------------------
 0.0    1.0   -1426.6971 -1453.1748 -1400.2194  True 
 0.0    2.0   -1096.3589 -1120.8581 -1071.8598  True 
 0.0    3.0    122.1555   97.9424    146.3686   True 
 1.0    2.0    330.3382   305.7966   354.8797   True 
 1.0    3.0   1548.8526  1524.5967  1573.1086   True 
 2.0    3.0   1218.5145  1196.4353  1240.5937   True 
-----------------------------------------------------
'''
'''
结论：从整体上看，不同周数销量存在显著差异，在第18周以前，偶数周的销量普遍高于奇数周；而在19周以后，奇数周的销量普遍高于偶数周。思考：根据这一分布特性定义新特征，18周以前且为偶数周（0），18周以前且为奇数周（1），19周以后且为偶数周（2），19周以后且为奇数周（3）
新特征：WeekOfYearStr
'''

# 分析DayOfYear
df_DayOfYear = dataSet_3.groupby("DayOfYear", as_index=False)[["Sales"]].median()
'''
a = gbj.Scatter(x=df_DayOfYear.DayOfYear, y=df_DayOfYear.Sales, opacity=1)
data = [a]
layout = dict(title="DayOfYear")
fig = dict(data=data, layout=layout)
plotly.offline.plot(fig)
'''
q1, q3 = np.percentile(df_DayOfYear.Sales, q=[25,75])
q_delta = q3 - q1
proba = 0.5
low = q1 - proba*q_delta
high = q3 + proba*q_delta
df_DayOfYear["DayOfYearOutlier"] = 0
df_DayOfYear.loc[df_DayOfYear.Sales > high, "DayOfYearOutlier"] = 1
df_DayOfYear.loc[df_DayOfYear.Sales < low, "DayOfYearOutlier"] = 2
df_DayOfYear.DayOfYearOutlier.value_counts()
# 添加新特征
dataSet_3 = pd.merge(dataSet_3, df_DayOfYear[["DayOfYear","DayOfYearOutlier"]], on="DayOfYear", how="left")
df_test_open = pd.merge(df_test_open, df_DayOfYear[["DayOfYear","DayOfYearOutlier"]], on="DayOfYear", how="left")
df_test_open.DayOfYearOutlier.value_counts()
# 验证
mean_compare_DayOfYear = fd.oneway_anova(dataSet_3, x="DayOfYearOutlier", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   DayOfYearOutlier        Sales
0                 0  6623.739184
1                 1  9572.408229
2                 2  4675.975928

  Multiple Comparison of Means - Tukey HSD,FWER=0.05 
=====================================================
group1 group2  meandiff    lower      upper    reject
-----------------------------------------------------
  0      1     2948.669  2925.3228  2972.0153   True 
  0      2    -1947.7633 -2034.2466 -1861.2799  True 
  1      2    -4896.4323 -4985.2868 -4807.5778  True 
-----------------------------------------------------
'''

df_DayOfYear_2 = dataSet_3.groupby("DayOfYear", as_index=False)[["Sales"]].median()
df_DayOfYear_2["SalesPreDay"] = df_DayOfYear_2.Sales.shift(1)
df_DayOfYear_2 = df_DayOfYear_2.fillna(value=3850)
df_DayOfYear_2["DayOfYearSlope"] = df_DayOfYear_2.Sales - df_DayOfYear_2.SalesPreDay

q1, q3 = np.percentile(df_DayOfYear_2.DayOfYearSlope, q=[25,75])
q_delta = q3 - q1
proba = 1.0
low = q1 - proba*q_delta
high = q3 + proba*q_delta
df_DayOfYear_2["DayOfYearSlopeStr"] = 0
df_DayOfYear_2.loc[df_DayOfYear_2.DayOfYearSlope > high, "DayOfYearSlopeStr"] = 1
df_DayOfYear_2.loc[df_DayOfYear_2.DayOfYearSlope < low, "DayOfYearSlopeStr"] = 2
df_DayOfYear_2.DayOfYearSlopeStr.value_counts()
# 添加新特征
dataSet_3 = pd.merge(dataSet_3, df_DayOfYear_2[["DayOfYear","DayOfYearSlopeStr"]], on="DayOfYear", how="left")
df_test_open = pd.merge(df_test_open, df_DayOfYear_2[["DayOfYear","DayOfYearSlopeStr"]], on="DayOfYear", how="left")
df_test_open.DayOfYearSlopeStr
mean_compare_DayOfYear_2 = fd.oneway_anova(dataSet_3, x="DayOfYearSlopeStr", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   DayOfYearSlopeStr        Sales
0                  0  6790.572084
1                  1  8631.813294
2                  2  7088.488833

  Multiple Comparison of Means - Tukey HSD,FWER=0.05 
=====================================================
group1 group2  meandiff    lower      upper    reject
-----------------------------------------------------
  0      1    1841.2412  1812.7206  1869.7618   True 
  0      2     297.9167   261.2328   334.6007   True 
  1      2    -1543.3245 -1588.2565 -1498.3924  True 
-----------------------------------------------------
'''
'''
结论：从整体上看，部分天存在高销量，部分天处于低谷；有些天前后出现飙升，而有些出现巨大落差。因此思考，新增四类日期数据，分别为“高峰日期”、“低谷日期”、“飙升日期”、“陡降日期”。“高峰”、“低谷”考虑极值，具体为越过0.5倍四分卫极差定义为高峰或低谷；“飙升”、“陡降”考虑前后两天斜率变化，具体为斜率越过1倍四分卫极差定义为飙升或陡降。
新特征：DayOfYearOutlier，DayOfYearSlopeStr
'''

# 分析Tenday
mean_compare_Tenday = fd.oneway_anova(dataSet_3, x="Tenday", y="Sales", W=10, H=8, use_hsd=True, plot=False)
'''
   Tenday        Sales
0       0  7286.921051
1       1  6844.829861
2       2  6755.968218

Multiple Comparison of Means - Tukey HSD,FWER=0.05
==================================================
group1 group2  meandiff   lower     upper   reject
--------------------------------------------------
  0      1    -442.0912 -461.5732 -422.6092  True 
  0      2    -530.9528 -550.3565 -511.5492  True 
  1      2     -88.8616 -108.0138  -69.7095  True 
--------------------------------------------------
'''
'''
结论：新增构建每月上中下三旬。从整体上看,每月的上中下三旬销量存在显著性差异。上旬日均7286，中旬6844，下旬6755，方差分析差异显著
'''

# 分析InPromo2
mean_compare_InPromo2 = fd.two_sample_ttest(dataSet_3, x="InPromo2", y="Sales", val_1=0, val_2=1, W=10, H=8, plot=False)
'''
   InPromo2        Sales
0         0  7263.266647
1         1  6547.443950
'''
'''
结论：从整体数据来看，处在长期促销时期门店的销量反而更低
'''

# 分析特殊日期
mean_compare_WillClosedTomorrow_TodayIsSat = fd.two_sample_ttest(dataSet_3, x="WillClosedTomorrow_TodayIsSat", y="Sales", val_1=0, val_2=1, W=10, H=8, plot=False)
'''
   WillClosedTomorrow_TodayIsSat        Sales
0                              0  7179.393519
1                              1  5836.286592
'''
mean_compare_WillClosedTomorrow_TodayIsNotSat = fd.two_sample_ttest(dataSet_3, x="WillClosedTomorrow_TodayIsNotSat", y="Sales", val_1=0, val_2=1, W=10, H=8, plot=False)
'''
   WillClosedTomorrow_TodayIsNotSat        Sales
0                                 0  6937.021605
1                                 1  7734.233285
'''
mean_compare_WasClosedYesterday_TodayIsMon = fd.two_sample_ttest(dataSet_3, x="WasClosedYesterday_TodayIsMon", y="Sales", val_1=0, val_2=1, W=10, H=8, plot=False)
'''
   WasClosedYesterday_TodayIsMon        Sales
0                              0  6727.049829
1                              1  8169.791028
'''
mean_compare_WasClosedYesterday_TodayIsNotMon = fd.two_sample_ttest(dataSet_3, x="WasClosedYesterday_TodayIsNotMon", y="Sales", val_1=0, val_2=1, W=10, H=8, plot=False)
'''
   WasClosedYesterday_TodayIsNotMon        Sales
0                                 0  6934.039577
1                                 1  7603.970607
'''
'''
结论：明日闭店且今日是周六（WillClosedTomorrow_TodayIsSat）、明日闭店且今日不是周六（WillClosedTomorrow_TodayIsNotSat）、昨日闭店且今日是周一（WasClosedYesterday_TodayIsMon）、昨日闭店且今日不是周一（WasClosedYesterday_TodayIsNotMon）。分析中发现这四类特殊的日期下平均销量较为异常，不是异常高就是异常低。因此考虑单独标记作为新特征
'''

# 保存数据
f = open(txt_path + "dataSet_3.txt", "wb"); pickle.dump(dataSet_3, f); f.close() # 仅开业有销量数据
f = open(txt_path + "df_test_open.txt", "wb"); pickle.dump(df_test_open, f); f.close() # 测试集仅开业数据
