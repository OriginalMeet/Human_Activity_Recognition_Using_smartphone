import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# train dataset's X and y
X_train = np.loadtxt("train/X_train.txt")
y_train = np.loadtxt("train/y_train.txt")

# test dataset's X and y
X_test = np.loadtxt("test/X_test.txt")
y_test = np.loadtxt("test/y_test.txt")

# doing preprocessing of the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# error =  Label must be in [0, 6), but found 6 in label
# solution = https://github.com/dmlc/xgboost/issues/1343
# in here to solve this problem I have to change the label which are 1,2,3,4,5,6 to 0,1,2,3,4,5

# changing the labels of y train
for i in range(len(y_train)):
    y_train[i] = y_train[i] - 1

# changing the labels of y test
for j in range(len(y_test)):
    y_test[j] = y_test[j] - 1

# importing the LightGBM library
import lightgbm

# training data for lightGBM
train_data_lgbm = lightgbm.Dataset(X_train,label=y_train)

############################################################################
############################################################################
# 10 times 10 fold for parameter tuning
lightgbm_classifier_cv = lightgbm.LGBMClassifier(    
    application= 'multiclass',
    objective= 'multiclass',
    metric= 'multi_logloss',
    num_class=6,
    is_unbalance= 'true',
    boosting= 'gbdt',
    num_leaves= 31,
    feature_fraction= 0.5,
    bagging_fraction= 0.5,
    bagging_freq= 20,
    learning_rate= 0.05
)
scores = []
for i in range(10):
    scores.append(cross_val_score(lightgbm_classifier_cv,X_train,y_train,cv=10))

print("The 10 times 10 fold accuracy of the lightGBM classifier is: ",np.average(scores))   

############################################################################
############################################################################
# parameters for the light gbm
parameters = {
    'application': 'multiclass',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class':6,
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

# creating the model of the lightgbm
model = lightgbm.train(parameters,train_data_lgbm,num_boost_round=5000)

# predicting the accuracy
y_pred=model.predict(X_test)
y_prd_final = np.argmax(y_pred,axis=-1)

# printing accuracy of lightGBM
print("the accuracy of LightGBM: ",accuracy_score(y_test,y_prd_final))

# confusion metrics of the LightGBM
confusion_matrix_LightGBM = confusion_matrix(y_test,y_prd_final)
print(confusion_matrix_LightGBM)