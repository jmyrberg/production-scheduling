# -*- coding: utf8 -*-
'''
Created on 15.1.2017

@author: Jesse Myrberg / jesse.myrberg@gmail.com
'''
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from statsmodels.tsa import arima_model as am

class MedianForecaster(object):
    '''Predicts electricity price values based on median (mean) for the same weeks of previous years.
    
    For example, 29.9.2016 23:00:00 (Thursday of week 201639) is forecasted based on the prices 
    of 24.9.2015 23:00:00 (Thursday of week 201539) and 25.9.2014 23:00:00 (Thursday of week 201539).'''
    
    def __init__(self):
        self.name = 'MedianForecaster'
        self.d = None
    
    def fit(self,X,y):
        '''Fits previous values in model.
        
        X: DateTimeIndex
        y: Values to train on'''
        ts = pd.Series(y,index=X,name='value')
        #d = ts.groupby(by=[ts.index.month,ts.index.day,ts.index.hour]).median().to_dict()
        d = ts.groupby(by=[ts.index.week,ts.index.weekday,ts.index.hour]).median().to_dict()
        # If (week,weekday) -combination doesn't exist, return median
        self.d = defaultdict(lambda: ts.iloc[:,1].median(),d) 
        return(self)
        
    def predict(self,X):
        '''Predicts values in X.
        
        X: DateTimeIndex for those timestamps that need to be predicted'''
        #res = np.array([self.d[(month,day,hour)] for month,day,hour in zip(X.month,X.day,X.hour)])
        res = np.array([self.d[(week,weekday,hour)] for week,weekday,hour in zip(X.week,X.weekday,X.hour)])
        return(res)

class SARIMA(object):
    '''SARIMA -time series model for electricity price prediction (work in progress).'''
    
    def __init__(self):
        self.name = 'SARIMA'
        self.model = None
    
    def fit(self,X,y,p=4,d=1,q=24,**kwargs):
        '''Info on model parameters etc:
        http://statsmodels.sourceforge.net/0.6.0/tsa.html'''
        self.model = am.ARIMA(y, order=(p,d,q), exog=None, dates=X, freq='H', missing='none')
        self.model = self.model.fit(start_params=None, trend='c', method='css-mle', 
                       transparams=True, solver='lbfgs', maxiter=50, 
                       full_output=1, disp=5, callback=None,**kwargs)
        self.start_n = X.shape[0]#X.max() + pd.DateOffset(hours=1)
        print(self.model.summary())
        return(self)
    
    def predict(self,X):
        res = self.model.predict(start=self.start_n, end=self.start_n+100, dynamic=True)
        plt.plot(res)
        plt.show()
        print(res)