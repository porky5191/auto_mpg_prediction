# This will act as file so that we can import anything whenever we want. This will act as a package

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# -------- Functions -------------

## preprocess custom original column in data
def preprocess_origin_cols(df):
    '''
    Replace the values of origin columns
        1 is replaced by 'India'
        2 is replaced by 'USA'
        3 is replaced by 'Germany'
    '''
    df['Origin'] = df['Origin'].map({1:'India', 2:'USA', 3:'Germany'})
    return df

# Creating custom attribute adder class
acc_ix, hpower_ix, cyl_ix = 4, 2, 0
class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True):
        self.acc_on_power = acc_on_power # no *args or **kwargs
    def fit(self, x, y=None):
        return self # nothing else to do
    def transform(self, x):
        acc_on_cyl = x[:, acc_ix]/x[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = x[:, acc_ix] / x[:, hpower_ix]
            return np.c_[x, acc_on_power, acc_on_cyl] # np.c_ will concate numpy array
        return np.c_[x, acc_on_cyl]

# Pipeline
def num_pipeline_transformer(data):
    '''
    Function to process numerical transformations
    Argument:
         data: original dataframe
    Returns:
         num_attrs: numerical dataframe
         num_pipeline: numerical pipeline object
    '''
    
    numerics = ['float64', 'int64']
    
    num_attrs = data.select_dtypes(include=numerics)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler())
    ])
    data.head()
    return num_attrs, num_pipeline

# Full Transformer
def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both
    numerical and categorical data.
    
    Argument:
        data: original dataframe
    Returns:
        prepared_data: transformed data, ready to use
    '''
    cat_attr = ['Origin']
    num_attrs, num_pipeline = num_pipeline_transformer(data)

    full_pipeline =ColumnTransformer([
        ('num', num_pipeline, num_attrs.columns),
        ('cat', OneHotEncoder(), cat_attr)
    ])
    
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data


def predict_mpg(config, model):
    '''
    This will predict the output of a model
    Arguments:
       a. config - our data which is either in dictionary or dataframe formate
       b. model - our ML model which is used to predict the output
    Return: 
       returns an array of predicted value
    '''
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    preproc_df = preprocess_origin_cols(df)
    prepared_df = pipeline_transformer(preproc_df)
    y_pred = model.predict(prepared_df)
    return y_pred