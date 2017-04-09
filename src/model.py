# -*- coding: utf8 -*-
'''
Created on 15.1.2017

@author: Jesse Myrberg / jesse.myrberg@gmail.com
'''
import numpy as np
import pandas as pd
from data_utils import load_data, itemnames, split_prices,\
    format_forecast_results, prepare_optimization, get_existing_schedule_info
from forecast import MedianForecaster
from optimization import create_optimization_problem, solve_problem
from data_exploration import plot_schedule, plot_forecasted_schedule

def MSE(act,pred):
    '''Mean Squared Error for forecast evaluation.'''
    return(np.sum(np.power(act-pred,2)) / len(act))
    
def run_realized_schedule():
    '''Information and plot for realized schedule.'''
    schedule = get_existing_schedule_info()
    plot_schedule(schedule,title='Realized schedule')
    
def run_theoretical_optimum():
    '''Solve optimization problem using realized forecast.'''
    
    # Load data
    _,items,schedule = load_data()
    
    # Solve optimization problem with actual electricity prices
    itemblocks_to_produce = schedule[itemnames()].sum(0).to_dict()
    blocks_available = pd.unique(schedule.blockid)
    block_order = blocks_available
    actual_block_prices = schedule.set_index('blockid').price.to_dict()
    item_consumptions = items.set_index('item').consumption.to_dict()
    prob = create_optimization_problem(itemblocks_to_produce,blocks_available,actual_block_prices,item_consumptions)
    schedule = solve_problem(prob,actual_block_prices,item_consumptions,block_order)
    
    plot_forecasted_schedule(schedule,title='Optimal schedule')

def run_model(model):
    '''Solve optimization problem using MedianForecaster -model for forecasting.'''
    
    # Load data
    prices,items,schedule = load_data()
    
    # Split to train and test sets based on realized schedule
    X_train,y_train,X_test,y_test = split_prices(prices)
    
    # Fit and predict with model
    y_pred = model.fit(X_train,y_train).predict(X_test)
    
    # Evaluate model
    mse = MSE(y_test,y_pred)
    print('MSE: %f' % mse)
    
    # Solve optimization problem with forecasted electricity prices
    df_pred = format_forecast_results(X_test,y_pred,prices,schedule)
    df_pred.plot()
    itemblocks_to_produce,blocks_available,forecasted_block_prices, \
        actual_block_prices,item_consumptions,block_order = prepare_optimization(items,schedule,df_pred)
    prob = create_optimization_problem(itemblocks_to_produce, blocks_available, forecasted_block_prices, item_consumptions,
                                name=model.name)
    solution_schedule = solve_problem(prob, actual_block_prices, item_consumptions, block_order)
    
    plot_forecasted_schedule(solution_schedule,title='Schedule using %s' % model.name)

def main():
    print('\n-- Realized schedule')
    run_realized_schedule()
    print('\n-- Theoretical optimum schedule')
    run_theoretical_optimum()
    print('\n-- MedianForecaster based schedule')
    run_model(MedianForecaster())

if __name__ == '__main__':
    main()
