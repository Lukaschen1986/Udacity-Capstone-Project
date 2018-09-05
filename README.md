# Udacity-Capstone-Project

#### 开发环境与必备库
###### Python 3.6.6
###### pickle, numpy, pandas, scipy, statsmodels, sklearn, xgboost, matplotlib, seaborn, plotly

#### rawData：原始数据集  
###### train.csv  
###### test.csv  
###### store.csv  
###### sample.csv  
###### sample_submission.csv  
  
#### dataSet：建模过程性数据  
###### dataSet_1.txt：第一次特征工程后数据  
###### dataSet_2.txt：第二次特征工程后数据  
###### dataSet_3.txt：训练数据集（仅开业）  
###### df_test_open_1.txt：单体模型预测结果  
###### df_test_open_2.txt：整体模型预测结果  
###### df_test_closed.txt：测试集仅闭店数据  

#### script：模型脚本
###### 1-data_process.py：数据预处理（10分钟）
###### 2-data_explore.py：探索性数据分析（30分钟）
###### 3-model_on_singlestore.py：单体模型（1.5天）
###### 4-model_on_allstores.py：整体模型（1.5小时）
###### 5-model_merge.py：模型融合（1分钟）
###### func_detail.py：具体函数

#### result：预测结果
###### df_rmspe_singlestore.csv：单体模型每个店铺验证集rmspe  
###### df_rmspe_allstores.csv：整体模型验证集rmspe
###### df_submission：提交结果

#### paper：毕业论文
