# importing libraries for the catboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score # to get the accuracy score from predicted and real value
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import cross_val_score

# importing train dataset
X_train = np.loadtxt("train/X_train.txt")
y_train = np.loadtxt("train/y_train.txt")

# importing the test dataset
X_test = np.loadtxt("test/X_test.txt")
y_test = np.loadtxt("test/y_test.txt")

# doing pre-processing of the train and test dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#########################################################
# performing cross validation to find the best parameters.
#########################################################
# catboost classifier
cat_cv_classifier = CatBoostClassifier(
    iterations = 1000, # 1000 are ideal
    loss_function='MultiClass',
    bootstrap_type = "Bayesian",
    eval_metric = 'MultiClass',
    leaf_estimation_iterations = 100,
    random_strength = 0.5,
    depth = 7,
    l2_leaf_reg = 5,
    learning_rate=0.1,
    bagging_temperature = 0.5,
    task_type = "GPU",
)
# cross validation function to find the best parameters
scores = []
for i in range(10):
    scores.append(cross_val_score(cat_cv_classifier,X_train,y_train,cv=10))
print(scores)
print("The 10 times 10 fold accuracy of the catboost classifier is: ",np.average(scores))    

#########################################################
# creating a catboosting classifier with optimized parameters and testing the same on test dataset
#########################################################
cat_model = CatBoostClassifier(
    iterations = 1000, # 1000 are ideal
    loss_function='MultiClass',
    bootstrap_type = "Bayesian",
    eval_metric = 'MultiClass',
    leaf_estimation_iterations = 100,
    random_strength = 0.5,
    depth = 7,
    l2_leaf_reg = 5,
    learning_rate=0.1,
    bagging_temperature = 0.5,
    task_type = "GPU",
)

# training the model
cat_model.fit(X_train,y_train)

# predicting the model output
y_pred_cat = cat_model.predict(X_test)
# printing the accuracy of the tuned model
print("accuracy of the catboost: ",accuracy_score(y_test,y_pred_cat))

# confusion metrics of the LightGBM and plotting the same
confusion_matrix_LightGBM = confusion_matrix(y_test,y_pred_cat)
print(confusion_matrix_LightGBM)    