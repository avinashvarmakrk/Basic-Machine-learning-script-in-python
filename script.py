# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 13:37:29 2016

@author: Avinash
"""

#importing required modules
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error






#reading files
train = pd.read_csv("codetest_train.txt", sep = "\t")
test = pd.read_csv("codetest_test.txt", sep = "\t")

#summarizing the data
train.describe()


#target is a conintuous variable. Hence the task is a regression problem and regression algorithms will be used

#checking for missing values

#for train
max(5000 - train.count())   #maximum 132 missing values in a column
np.mean(5000 - train.count()) #on an average 99 missing values per data which is less compared to the 5000 entries
train.dropna(thresh=200) #dropping rows with less than 200 non missing values
train.dtypes #all columns are either float64 or objecct type

#for test
max(1000 - test.count())   #maximum 34 missing values in a column ie 3.4%
np.mean(1000 - test.count()) #on an average 19 missing values per data which is less compared to the 1000 entries
test.dropna(thresh=200) #dropping rows with less than 200 non missing values
test.dtypes #all columns are either float64 or objecct type


#imputing missing values
#if object type, filling using "missing" else filling using mean
def missing_handler(arr):
    for column in arr:
        if arr[column].dtype == "object":
            arr[column].fillna("missing",inplace=True)
        else:
            arr[column].fillna(arr[column].mean(),inplace = True)
    return (arr)
    
train = missing_handler(train)  
test = missing_handler(test) 

#separating target variable from the rest of the data 
Y = train['target'].values
X = train.drop(['target'],axis=1)

#converting object types to categorial
for f in X.columns:
    if X[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X[f].values) + list(test[f].values))
        X[f] = lbl.transform(list(X[f].values))
        test[f] = lbl.transform(list(test[f].values))

#splitting training set for cross validation later
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X,Y,random_state=2016)

#only 3-fold cv being done due to time constraints

#constructing algorithms
#training random forest with optimum parameters using gridsearchCV
rfr = RandomForestRegressor(n_estimators = 500,max_depth = 15, n_jobs = -1, random_state = 2016, verbose = 1)
#parameters = {'n_estimators': [100,200,300,400,500], 'max_depth': [5,8,10,15,20,25,30]}
parameters = {'n_estimators':[100,200,300,400,500], 'max_depth': [6,8,12,15,20]}

model_rfr_allfeatures = grid_search.GridSearchCV(estimator =rfr, param_grid = parameters, n_jobs = -1, cv = 3, verbose = 20, scoring='mean_squared_error')
model_rfr_allfeatures.fit(X_train, Y_train)
print(model_rfr_allfeatures.best_params_) #max_depth = 15, n_estimatores = 500
print(model_rfr_allfeatures.best_score_)

predictions_rfr_allfeatures = model_rfr_allfeatures.predict(X_test)
mean_squared_error(Y_test, predictions_rfr_allfeatures)  #12.19




#training GBM with optimum parameters
gbr = GradientBoostingRegressor(loss='ls',random_state=2016)
parameters = {'learning_rate': [0.1,0.01], #when use hyperthread, xgboost may become slower
              #'max_depth': [6,15],
              'max_depth': [6,8,15,30],
              'n_estimators' : [100,200,400,500], 
              'max_features': ['auto'],
              #'num_class':3,
              }

        
model_gbr_allfeatures = grid_search.GridSearchCV(estimator =gbr, param_grid = parameters, n_jobs = -1, cv = 2, verbose = 20, scoring='mean_squared_error')
model_gbr_allfeatures.fit(X_train, Y_train)
print(model_gbr_allfeatures.best_params_) #'max_depth': 6, 'n_estimators': 500, 'learning_rate': 0.1, 'max_features': 'auto'

predictions_gbr_allfeatures = model_gbr_allfeatures.predict(X_test)
mean_squared_error(Y_test, predictions_gbr_allfeatures) #7.071566

gbr_final = GradientBoostingRegressor(loss='ls',random_state=2016,max_depth = 6, learning_rate = 0.1, n_estimators = 500, max_features = "auto")

final_model = gbr_final.fit(X,Y)
predictions = final_model.predict(test)

#feature importance
print(final_model.feature_importances_)
importance = final_model.feature_importances_
importance = pd.DataFrame(importance)
importance["Feature"] = test.columns.values
importance.columns = ["Importance","Feature"]
importance.sort_values(["Importance"],inplace = True, ascending = False)
importance.head()
importance.tail()
importance.to_csv('feature_importance.csv',index = False, header = True)

#creating the output file
predictions = pd.DataFrame(predictions)
predictions.to_csv('Output.txt', index=False,header = False)




#training svms with optimum parameters
svr = svm.SVR()
parameters = {'C':[1], 
              'cache_size':[4000],
              'coef0': [0],
              'degree': [3],
              'epsilon': [0.2],
              'gamma': ['auto'],
              'kernel': ['linear','rbf'],
              'tol':[0.1,0.01,0.001],
              }
model_svm_allfeatures = grid_search.GridSearchCV(estimator =svr, param_grid = parameters, n_jobs = -1, cv = 2, verbose = 20, scoring='mean_squared_error')
model_svm_allfeatures.fit(X_train, Y_train)
print(model_svm_allfeatures.best_params_) #'degree': 3, 'C': 1, 'gamma': 'auto', 'kernel': 'linear', 'tol': 0.1, 'epsilon': 0.2, 'coef0': 0, 'cache_size': 2000
print(model_svm_allfeatures.best_score_)

predictions_svm_allfeatures = model_svm_allfeatures.predict(X_test)
mean_squared_error(Y_test, predictions_svm_allfeatures) #12.9097


##############################################################################
#feature selection

#PCA to remove correlations
from sklearn import decomposition
pca = decomposition.PCA(n_components=50)
pca.fit(X)
X_PCA = pca.transform(X)
test_PCA = pca.transform(test)
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X_PCA,Y,random_state=2016)

gbr = GradientBoostingRegressor(loss='ls',random_state=2016)
parameters = {'learning_rate': [0.1], #when use hyperthread, xgboost may become slower
              #'max_depth': [6,15],
              'max_depth': [6],
              'n_estimators' : [500], 
              'max_features': ['auto'],
              #'num_class':3,
              }

        
model_gbr_allfeatures = grid_search.GridSearchCV(estimator =gbr, param_grid = parameters, n_jobs = -1, cv = 2, verbose = 20, scoring='mean_squared_error')
model_gbr_allfeatures.fit(X_train, Y_train)
print(model_gbr_allfeatures.best_params_) #'max_depth': 6, 'n_estimators': 500, 'learning_rate': 0.1, 'max_features': 'auto'


predictions_gbr_allfeatures = model_gbr_allfeatures.predict(X_test)
mean_squared_error(Y_test, predictions_gbr_allfeatures) #7.071566



#ensembling randomForest model using bagging
bag = BaggingRegressor(rfr, n_estimators=500, max_samples=0.1, random_state=25)
bag.fit(X_train, Y_train)
predictions_rfr_bagging = bag.predict(X_test)
mean_squared_error(Y_test, predictions_rfr_bagging)

#recursive selection of features for randomForest
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=rfr, step=1, cv=3,
              scoring='mean_squared_error')
rfecv.fit(X_train, Y_train)
print("Optimal number of features : %d" % rfecv.n_features_) 




                     
