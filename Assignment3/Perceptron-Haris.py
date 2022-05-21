
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('train.csv')
df_train.head()
df_test = pd.read_csv('test.csv')
df_test.head()
df_testing = pd.read_csv('test.csv')
#df_testing.head()
print(df_train.isnull().sum())
print(df_test.isnull().sum())
df_train.describe()
df_train.drop(['id'],axis=1,inplace=True)
df_test.drop(['id'],axis=1,inplace=True)
df_train.hist(figsize=(15, 15))
df_test.hist(figsize=(15, 15))
print(df_train.info())
print(df_test.info())
df_train.skew()
df_test.skew()
df_train.corr()
corr = df_train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="inferno", square=True)
corr = df_test.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="inferno", square=True)
X=df_train.drop(['target'],axis=1)
y=df_train.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.18, random_state=2)
import lightgbm as lgb
LGB = lgb.LGBMRegressor(random_state=30, n_estimators=540, min_data_per_group=5, boosting_type='gbdt',
 num_leaves=256, max_dept=-2, learning_rate=0.005, subsample_for_bin=200000,
 lambda_l1= 1.074622455507616e-05, lambda_l2= 2.0521330798729704e-06, n_jobs=-1, cat_smooth=1.0, 
 importance_type='split', metric='rmse', min_child_samples=20)

LGB.fit(X_train, y_train)
pred_LGB = LGB.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse_LGB = np.sqrt(mean_squared_error(y_test, pred_LGB))
rmse_LGB
pred_LGB = LGB.predict(df_test)
pred_LGB
output = pd.DataFrame({'id': df_testing.id, 'target': pred_LGB})
output.to_csv('Kaggle_Playground_Submission_LGB.csv', index=False)
