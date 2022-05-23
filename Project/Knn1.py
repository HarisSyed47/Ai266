import pandas as pd 
import pickle 
import numpy as np
from sklearn import preprocessing 
from sklearn.metrics import accuracy 
from sklearn.model_selection import train 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

Nf = pd.read_csv("C:/Harispc//Downloads/train.csv")
Nf.shape
del Nf['id']
del Nf['k_27'] 
Nf.head(6)

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
X = Nf.drop(columns=['target'])
y = Nf['target']
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importance)
feat_importances = pd.Series(model.feature_importance, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()

newNf = Nf[["k_00","k_01","k_02","k_03","k_04","k_05","k_06","k_07","k_08","k_09","k_10",
"k_11","k_12","k_13","k_14","k_15","k_16","k_17","k_18","k_19","k_20","k_21","k_22","k_23","k_24","k_25",
"k_26","k_28","k_29","k_30","targets"]]
newNf.head(6)

XTT = newNf.drop(columns=['target'])
yTT = newNf['target']
X_train, X_test, y_train, y_test = train_test_split(XTT, yTT, test_size=0.00001)
modelKNN = KNeighborsClassifier(n_neighbors=77)
resultKNN = modelKNN.fit(X_train, y_train)
prediction_test = modelKNN.predict(X_test)
accKNN = metrics.accuracy(y_test, prediction_test)
print("KNN Accuracy: ", accKNN)

Nftest = pd.read_csv("C:/Harispc//Desktop/test.csv")
Nftest.head(6)

newNf = Nf[["k_00","k_01","k_02","k_03","k_04","k_05","k_06","k_07","k_08","k_09","k_10",
"k_11","k_12","k_13","k_14","k_15","k_16","k_17","k_18","k_19","k_20","k_21","k_22","k_23","k_24","k_25",
"k_26","k_28","k_29","k_30","targets"]]
newtest.head(6)

newCSV = Nftest[['id']]
newCSV

predictOnTest = modelKNN.predict(new)
predictOnTest

newCSV['target'] = predictOnTest 
newCSV

newCSV.to_csv('Output.csv', index=False)

print("Accuracy is: ", acc)
