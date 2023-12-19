from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd



class FillingNaN(BaseEstimator, TransformerMixin):

    def __init__(self, fill_value=0) -> None:
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = np.where(pd.isna(X) | (X == np.nan), self.fill_value, X)
        return X_
    



class StringToNumber(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = np.where(X == 'one', 1,
                      np.where(X == 'two', 2,
                               np.where(X == 'three', 3,
                                        np.where(X == 'four', 4,
                                                 np.where(X == 'five', 5,
                                                          np.where(X == 'six', 6,
                                                                   np.where(X == 'seven', 7,
                                                                            np.where(X == 'eight', 8,
                                                                                     np.where(X == 'nine', 9,
                                                                                              np.where(X == 'ten', 10, 
                                                                                                       np.where(X == 'eleven', 11, X)
                                                                                             )
                                                                                    )
                                                                           )
                                                                  )
                                                         )
                                                )
                                       )
                              )
                    )
        )
        return X_       





def pipeLines():

    tran1 = ColumnTransformer([
        ('fillna', FillingNaN(), [0]),
    ], remainder='passthrough')


    tran2 = ColumnTransformer([
        ('str_to_num', StringToNumber(), [0]),
    ], remainder='passthrough')

    tran3 = ColumnTransformer([
        ('simple_impute', SimpleImputer(), [1, 2])
    ], remainder='passthrough')

    tran4 = LinearRegression()

    pipeline = make_pipeline(tran1, tran2, tran3, tran4)

    return pipeline

