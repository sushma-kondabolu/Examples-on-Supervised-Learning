
"""
Created on Thu Jun  3 20:10:14 2021

@author: ksnss
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


housing_data=pd.read_csv("C:/Users/ksnss/Downloads/housing.csv")

df_calhousingdata=pd.DataFrame(housing_data,columns=housing_data.columns)

#question1
def summary_stats(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.describe())

#question2
def correlation_median_house_value(df):
    
    df_calhousingdata['ocean_proximity'] =df_calhousingdata['ocean_proximity'].astype('category').cat.codes
    sns.pairplot(data=df,y_vars=['median_house_value'], x_vars=df.columns,diag_kind = None)
    print(df[df.columns[:]].corr()['median_house_value'][:])



#question 4

def cleaning_method_missing_values(df):
    series=df.isnull().sum()
    print("Before replacing the missing values for the dataframe in all columns: ")
    print(series)
    columns=df.columns
    for col in columns:
        if df[col].isnull().sum():
            df[col].fillna(df[col].mean(), inplace = True)
    print("After replacing the missing values for the dataframe in all columns: ")
    print(df.isnull().sum())
    return df


def ordinal_encoding_ocean_proximity(df):
    print('Data sample before Ordinal Encoding')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.head(5))
    enc=OrdinalEncoder()
    arr=df[['ocean_proximity']]
    arr_enc=enc.fit_transform(arr)
    df['ocean_proximity']=arr_enc
    print('Data sample after Ordinal Encoding')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.head(5))
    return df


def standard_scaler(df):
    print('Data sample before Standard Scaling')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.head(5))
    scaler=StandardScaler()
    df=scaler.fit_transform(ordinal_encoding_ocean_proximity(df))
    scaled_df=pd.DataFrame(df,columns=housing_data.columns)
    print('Data sample after Standard Scaling')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(scaled_df.head(5))
    return scaled_df
#add df.head() for displaying sample data #display all unique values for

#question5
def new_features(df):
    df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]
    df["rooms_per_household"] = df["total_rooms"]/df["households"]
    df["population_per_household"]=df["population"]/df["households"]
    print('Data sample after adding new features')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.head(5))
    return df

#question 6
#linear regression using math
def closed_form_solution(X,y):
    w=np.linalg.inv(X.T@X)@X.T@y
    return w



def linear_regression(df):
    df1=df.drop("median_house_value",axis=1)
    X_train,X_test,y_train,y_test=train_test_split(df1,df["median_house_value"],test_size=0.2,random_state=0)
    X_train_b=np.column_stack((np.ones(len(X_train)),X_train))
    w=closed_form_solution(X_train_b,y_train)
    print(w)
    
    y_train_pred=X_train_b @ w
    rmse=np.sqrt(MSE(y_train, y_train_pred))
    r2=r2_score(y_train,y_train_pred)
    print('RMSE(Train) :',rmse)
    print('R2(Train):',r2)
    
    X_test_b=np.column_stack((np.ones(len(X_test)),X_test))
    y_test_pred=X_test_b @ w
    rmse=np.sqrt(MSE(y_test,y_test_pred))
    r2=r2_score(y_test,y_test_pred)
    print('RMSE(Test) :',rmse)
    print('R2(Test):',r2)

#linear regression using estimator
def lin_reg(df):
    df1=df.drop("median_house_value",axis=1)
    X_train,X_test,y_train,y_test=train_test_split(df1,df["median_house_value"],test_size=0.2,random_state=0)
    reg=LinearRegression()
    reg.fit(X_train,y_train)
    y_train_pred=reg.predict(X_train)
    rmse=np.sqrt(MSE(y_train, y_train_pred))
    r2=r2_score(y_train,y_train_pred)
    print('RMSE(Train) :',rmse)
    print('R2(Train):',r2)
    
    y_test_pred=reg.predict(X_test)
    rmse=np.sqrt(MSE(y_test,y_test_pred))
    r2=r2_score(y_test,y_test_pred)
    print('RMSE(Test) :',rmse)
    print('R2(Test):',r2)
    

#question7
def regularized_regression(df):
    alpha=0.00000001
    df1=df.drop("median_house_value",axis=1)
    X_train,X_test,y_train,y_test=train_test_split(df1,df["median_house_value"],test_size=0.2,random_state=0)
    n=0
    while n<10:
        alpha*=10
        reg=Ridge(alpha=alpha)
        reg.fit(X_train, y_train)
        y_train_pred=reg.predict(X_train)
        rmse=np.sqrt(MSE(y_train, y_train_pred))
        r2=r2_score(y_train,y_train_pred)
        print('Lambda value:',alpha)
        print('RMSE(Train) :',rmse)
        print('R2(Train):',r2)
        y_test_pred=reg.predict(X_test)
        rmse=np.sqrt(MSE(y_test,y_test_pred))
        r2=r2_score(y_test,y_test_pred)
        print('RMSE(Test) :',rmse)
        print('R2(Test):',r2)
        n=n+1

#question8
def Decision_tree(df):
    df1=df.drop("median_house_value",axis=1)
    X_train,X_test,y_train,y_test=train_test_split(df1,df["median_house_value"],test_size=0.2,random_state=0)
    dtr_rg=DecisionTreeRegressor(max_depth=9)
    dtr_rg.fit(X_train,y_train)
    y_train_pred=dtr_rg.predict(X_train)
    y_test_pred=dtr_rg.predict(X_test)
    rmse1=np.sqrt(MSE(y_train,y_train_pred))
    r21=r2_score(y_train,y_train_pred)
    print('RMSE(train) :',rmse1)
    print('R2(Train):',r21)
    rmse=np.sqrt(MSE(y_test,y_test_pred))
    r2=r2_score(y_test,y_test_pred)
    print('RMSE(Test) :',rmse)
    print('R2(Test):',r2)
    
summary_stats(df_calhousingdata)
#correlation_median_house_value(df_calhousingdata)
print(Decision_tree(standard_scaler(ordinal_encoding_ocean_proximity(cleaning_method_missing_values(new_features(df_calhousingdata)))))) 

print(regularized_regression(standard_scaler(ordinal_encoding_ocean_proximity(cleaning_method_missing_values(new_features(df_calhousingdata))))))

correlation_median_house_value(new_features(df_calhousingdata))

standard_scaler(df_calhousingdata)

print(lin_reg(standard_scaler(ordinal_encoding_ocean_proximity(cleaning_method_missing_values(new_features(df_calhousingdata))))))