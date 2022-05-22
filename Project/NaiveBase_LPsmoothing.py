import pandas as pd
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

train_cd = pd.read_csv ('/content/train.csv')

display(train_cd)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_cd=pd.read_csv('/content/train.csv')
#cleaning datasets
train_cd = pd.DataFrame(train_df)
train_cd['f_27'] = pd.to_numeric(train_df['f_27'], errors='coerce')
train_cd = train_cd.replace(np.nan, 0, regex=True)
y = train_cd.target
X = train_cd.drop('target', axis=1)
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.4)

import math
import random
import csv
 
 filename = r'G:\harispc\file..csv'
 
NaiveBasemodel = csv.reader(open(filename, "tm"))
NaiveBasemodel = list(NaiveBasemodel)
NaiveBasemodel = encode_class(NaiveBasemodel)
for i in range(len(Nbmodel)):
    NaiveBasemodel[i] = [float(x) for x in NaiveBasemodel[i]]
 
     

ratio = 0.4
train_data, test_data = splitting(NaiveBasemodel, ratio)
info = MeanAndStdDevForClass(train_data)
predictions = getPredictions(info, test_data)
accuracy = accuracy_rate(test_data, predictions)
print("Accuracy: ", ac)

y_true = train['InChI'].values
y_pred = ['InChI=1S/H2O/h1H2'] * len(train)
score = get_score(y_true, y_pred)
print(score)

mode_concat_string = ''
for i in range(11):
    mode_string = train[f'InChI_{i}'].fillna('nan').mode()[0]
    if mode_string != 'nan':
        if i == 0:
            mode_concat_string += mode_string
        else:
            mode_concat_string += '/' + mode_string
print(mode_concat_string)

y_true = train['InChI'].values
y_pred = [mode_concat_string] * len(train)
score = get_score(y_true, y_pred)
print(score)
