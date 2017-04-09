# -*- coding: utf8 -*-
'''
Created on 14.1.2017

@author: Jesse Myrberg / jesse.myrberg@gmail.com
'''
import pandas as pd
import numpy as np

def itemnames():
    """Return all possible item names"""
    g = ['KIS_NA_39', 'VII_57', 'MX_48', 'MX_56', 'KIS_NA_42', 'VII_54',
       'MX_S_48', 'MX_S_52', 'MX_52', 'KIS_NA_45', 'KIS_NA_51', 'MIP_45',
       'MIP_49', 'MIP_52', 'MIP_plus_48', 'MIP_plus_51', 'MX_42', 'MX_45',
       'MIP_G_42', 'KIS_42', 'KIS_NA_48']
    return(g)

def load_data(data_path='./data/'):
    """Load prices, items and realized schedule datasets"""
    # Electricity prices for 1.1.2014 - 14.10.2016 [e/MWh]
    prices = pd.read_csv(data_path + 'prices.csv', sep=";", decimal=",", names=['ts','blockid','price'], skiprows=1)
    prices['ts'] = pd.to_datetime(prices.ts, format='%d.%m.%Y %H:%M')
    prices.set_index('ts',inplace=True)
    
    # Paper item consumptions [MWh/block]
    items = pd.read_csv(data_path + 'items.csv', sep=";", decimal=",", names=['item','consumption'], skiprows=1)
    
    # One realized schedule, 75 blocks for 29.9.2016 - 11.10.2016
    schedule = pd.read_csv(data_path + 'schedule.csv', sep=";", decimal=",")
    schedule.columns = ['ts','blockid'] + itemnames() + ['price']
    schedule['ts'] = pd.to_datetime(schedule.ts, format='%d.%m.%Y %H:%M')
    return(prices,items,schedule)

def split_prices(prices,min_date='2016-09-29',max_date='2016-10-12'):
    """Split electricity prices to a date range (min_date,max_date)"""
    prices = prices["price"]
    
    X_train = prices[prices.index < min_date]
    X_train,y_train = X_train.index,X_train.values
    
    X_test = prices[(min_date <= prices.index) & (prices.index < max_date)]
    X_test,y_test = X_test.index,X_test.values
    return(X_train,y_train,X_test,y_test)

def format_forecast_results(X_test,y_pred,prices,schedule):
    """Format electricity price forecast results into a dataframe"""
    d_pred = dict((k,v) for k,v in zip(X_test,y_pred))
    df_pred = pd.DataFrame.from_dict(d_pred,'index').sort_index() # df from dict
    df_pred.columns = ['forecasted_price']
    df_pred = df_pred.merge(prices[['blockid']],left_index=True,right_index=True,how='left') # add blockid
    block_order = pd.unique(df_pred.blockid) # preserver order
    df_pred = df_pred.groupby('blockid')[['forecasted_price']].mean().reindex(block_order) # calculate block mean
    df_pred = df_pred.merge(schedule[['blockid','price']], left_index=True, right_on='blockid', how='left') \
                    .set_index('blockid').dropna() # add original price and drop mismatches
    return(df_pred)

def prepare_optimization(items,schedule,df_pred):
    """Prepare optimization problem parameters from original data and electricity price prediction"""
    itemblocks_to_produce = schedule[itemnames()].sum(0).to_dict()
    blocks_available = schedule.blockid.unique()
    block_order = pd.unique(schedule.blockid)
    forecasted_block_prices = df_pred['forecasted_price'].to_dict()
    actual_block_prices = df_pred['price'].to_dict()
    item_consumptions = items.set_index('item').consumption.to_dict()
    return(itemblocks_to_produce,blocks_available,forecasted_block_prices,
           actual_block_prices,item_consumptions,block_order)
    
def get_existing_schedule_info():
    """Load realized schedule and prepare it for plotting"""
    _,items,schedule = load_data()
    schedule['consumption'] = schedule[itemnames()].apply(lambda x: x * items.set_index('item') \
                                                       .T[itemnames()].as_matrix().T.flatten(), axis=1).sum(1)
    schedule['cost'] = schedule.consumption * schedule.price
    schedule['cumulative_cost'] = schedule.cost.cumsum()
    schedule.replace(0,np.nan,inplace=True)
    schedule.set_index('blockid',inplace=True)
    return(schedule)
    