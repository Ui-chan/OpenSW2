#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

def sort_dataset(dataset_df):
    sort_df = dataset_df.sort_values(by='year',ascending=True)
    return sort_df

# sort_df 에 dataset_df.sort_values를 사용해서 year을 기준으로 ascending=True 오름차순 정렬을 하여 저장하였다. 
# 그리고 sort_df 반환


def split_dataset(dataset_df):
    train_df = dataset_df[:1718]
    test_df = dataset_df[1718:]
    
    X_train = train_df.drop(columns=['salary'])
    Y_train = train_df['salary'] * 0.001
    X_test = test_df.drop(columns=['salary'])
    Y_test = test_df['salary'] * 0.001
    
    return X_train, X_test, Y_train, Y_test

# 일단 문제에 나온거처럼 dataset_df[:1718] 한 것을 train_df에 저장하고 dataset_df[1718:] 한 것을 test_df에 저장.
# X_train = train_df.drop(columns=['salary’])과 Y_train = train_df['salary'] * 0.001 를 하여 salary column을 label로 사용하고 label 값에 0.001을 곱해 사용. Test set과 train set 모두 동일하게 진행. 
# 그리고 return X_train, X_test, Y_train, Y_test한다. 

def extract_numerical_cols(dataset_df):
    numerical_columns = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    numerical_df = dataset_df[numerical_columns]
    return numerical_df

# 해당 dataset_df에 있는 여러 column들 중에 numerical_columns들만 뽑아 numerical_df에 저장 한 뒤 리턴한다.

def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, Y_train)

    dt_predictions = dt_model.predict(X_test)
    return dt_predictions

# decision tree regressor를 사용하기 위해 from sklearn.tree import DecisionTreeRegressor
# dt_model = DecisionTreeRegressor()를 통해 모델을 초기화 진행을 먼저 한 뒤, 
# dt_model.fit을 통해 X_train과 Y_train을 사용하여 학습진행.
# dt_predictions = dt_model.predict(X_test) X_test를 통해 X_test를 입력으로 받아 예측 값을 dt_predictions에 저장.
# 그리고 dt_predicitions을 리턴한다. 


def train_predict_random_forest(X_train, Y_train, X_test):
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, Y_train)
    
    rf_predictions = rf_model.predict(X_test)
    return rf_predictions

# random forest regressor를 사용하기 위해 from sklearn.ensemble import RandomForestRegressor
# rf_model = RandomForestRegressor()를 통해 모델을 초기화 진행을 먼저 한 뒤, 
# rf_model을 통해 X_train과 Y_train을 사용하여 학습진행.
# rf_predictions = rf_model.predict(X_test) X_test를 통해 X_test를 입력으로 받아 예측 값을 rf_predictions에 저장.
# 그리고 rf_predictions을 리턴한다. 


def train_predict_svm(X_train, Y_train, X_test):
    svm_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVR())
    ])
    svm_model.fit(X_train, Y_train)
    
    svm_predictions = svm_model.predict(X_test)

    return svm_predictions

# 우선 Pipeline을 사용하기 위해 from sklearn.pipeline import Pipeline을 한다. 
# 그리고 StandardScaler와 SVM 모델을 포함하는 파이프라인 초기화를 진행한다. StandardScaler를 사용하기 위해 from sklearn.preprocessing import StandardScaler. 나머지는 위 decision_tree와 random_forest와 동일하게 진행된다.
# Fit을 통해 파이프라인을 학습시키고, predict을 통해 예측값을 저장하고 그것을 return 한다. 
# 여기서 우린 회귀 작업을 수행하므로 SVR을 사용하였다. SVR을 사용하기 위해 from sklearn.svm import SVR.

def calculate_RMSE(labels, predictions):
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    return rmse

# 실제값과 예측 값 사이의 평균 제곱을 구할거다.
# 일단 mean_squared_error(labels, predictions)을 mse에 저장한다.(from sklearn.metrics import mean_squared_error)
# 그리고 mse의 제곱근을 계산하여 rmse를 구한다. 그리고 rmse를 리턴. 


if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))


# In[ ]:




