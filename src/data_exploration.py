# -*- coding: utf8 -*-
'''
Created on 14.1.2017

@author: Jesse Myrberg / jesse.myrberg@gmail.com
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from data_utils import itemnames, load_data, get_existing_schedule_info

def plot_schedule(schedule,title=''):
    """Plot original schedule as heatmap"""
    
    # Schedule heatmap
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.05)
    fig.suptitle(title, fontsize=14, fontweight='bold', x=0.51)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,5])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    cbar_ax = fig.add_axes([.91, .2, .03, .4])
    cbar_ax.set_title('Cost [e]', fontsize=10)
    palette = sns.light_palette("green", as_cmap=True)
    
    # Electricity price over time
    plt.subplot(gs[0])
    x = np.arange(schedule.shape[0])
    plt.plot(schedule.cumulative_cost.values, color="darkgreen")
    ax1.set_title('Total cost: %de' % schedule.cumulative_cost[-1].round(), fontsize=14, y=1.1)
    ax1.set_ylabel('Cost of production [e]')
    ax1.set_xlim([0,x.max()])
    ax1.set_axis_bgcolor("white")
    ax1.yaxis.grid(True, which='major', color="grey", linestyle='-', alpha=0.2)
    ax1.axhline(y=0, color="grey", linestyle='-', alpha=0.2)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off') 
    
    # Scheduling
    plt.subplot(gs[1])
    schedule[itemnames()] = schedule[itemnames()].apply(lambda x: x * schedule.cost)
    g = sns.heatmap(schedule[itemnames()].T,
                    annot=False,
                    linewidths=0.1, linecolor='white', 
                    vmin=0, vmax=4000,#(schedule.cost.max()).round(),
                    cmap=palette, 
                    cbar_ax=cbar_ax,
                    cbar=True,
                    ax=ax2)
    g.set_xticklabels(labels=schedule.index,rotation=90)
    g.set_yticklabels(labels=itemnames(),rotation=0)
    ax2.set_axis_bgcolor((237/255,237/255,237/255))
    ax2.set_xlabel('Four-hour block')
    ax2.set_ylabel('Paper item')

    plt.show()
    
def plot_forecasted_schedule(forecasted_schedule,title=''):
    """Plot optimized, forecast based schedule"""
    
    # Combine forecast and actual data
    schedule = forecasted_schedule
    actual_schedule = get_existing_schedule_info()
    schedule = schedule.merge(actual_schedule[['cost','cumulative_cost']], 
                              left_index=True, right_index=True, how='left')
    schedule['cost_savings'] = schedule.cost_y - schedule.cost_x
    schedule['cumulative_cost_savings'] = schedule.cumulative_cost_y - schedule.cumulative_cost_x
    
    # Schedule heatmap
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.05)
    fig.suptitle(title, fontsize=14, fontweight='bold', x=0.512)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,5])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    cbar_ax = fig.add_axes([.91, .2, .03, .4])
    cbar_ax.set_title('Cost [e]', fontsize=10)
    palette = sns.light_palette("green", as_cmap=True)
    
    # Electricity price over time
    plt.subplot(gs[0])
    x = np.arange(schedule.shape[0])
    plt.plot(schedule.cumulative_cost_savings.values, color="darkgreen")
    ax1.set_title('Total cost: %de\nTotal cost savings: %de (%.2f %%)' % \
                  (schedule.cumulative_cost_x[-1].round(),
                   schedule.cumulative_cost_savings[-1],
                   (schedule.cumulative_cost_savings[-1]/schedule.cumulative_cost_y[-1]*100).round(2)),
                   fontsize=14, y=1.1)
    ax1.set_ylabel('Cumulative cost savings [e]')
    ax1.set_xlim([0,x.max()])
    ax1.set_ylim([-10000,40000])
    ax1.set_axis_bgcolor("white")
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.yaxis.grid(True, which='major', color="grey", linestyle='-', alpha=0.2)
    ax1.axhline(y=-10000, color="grey", linestyle='-', alpha=0.2)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off') 
    
    # Scheduling
    plt.subplot(gs[1])
    schedule[itemnames()] = schedule[itemnames()].apply(lambda x: x * schedule.cost_x.fillna(0))
    g = sns.heatmap(schedule[itemnames()].T,
                    annot=False,
                    linewidths=0.1, linecolor='white', 
                    vmin=0, vmax=4000,#schedule.cost_savings.max().round(),
                    cmap=palette, 
                    cbar_ax=cbar_ax,
                    cbar=True,
                    ax=ax2)
    g.set_xticklabels(labels=schedule.index,rotation=90)
    g.set_yticklabels(labels=itemnames(),rotation=0)
    ax2.set_axis_bgcolor((237/255,237/255,237/255))
    ax2.set_xlabel('Four-hour block')
    ax2.set_ylabel('Paper item')
    plt.show()
    
def plot_data():
    """Plot prices and items over time"""
    
    # Load data
    prices,items,_ = load_data()
    
    # Price trends for different time periods
    prices['year'] = prices.ts.dt.year
    prices['month'] = prices.ts.dt.month
    prices['dayofweek'] = prices.ts.dt.dayofweek
    prices['hour'] = prices.ts.dt.hour
    
    prices.groupby('year',as_index=False)['price'].aggregate({"mean_price":np.mean, 
                                                              "median_price":np.median}).plot(kind='line', x="year")
    prices.groupby('month',as_index=False)['price'].aggregate({"mean_price":np.mean, 
                                                               "median_price":np.median}).plot(kind='line', x="month")
    prices.groupby('dayofweek',as_index=False)['price'].aggregate({"mean_price":np.mean, 
                                                                   "median_price":np.median}).plot(kind='line', x="dayofweek")
    prices.groupby('hour',as_index=False)['price'].aggregate({"mean_price":np.mean, 
                                                              "median_price":np.median}).plot(kind='line', x="hour")
    sns.FacetGrid(prices.groupby(["dayofweek","hour"])["price"] \
        .aggregate({"mean_price":np.mean, "median_price":np.median}).reset_index(), row="dayofweek") \
        .map(plt.plot, "hour", "median_price") \
        .set(xlim=(0, 23))
        
    # Grades
    item_order = items.loc[items.consumption.argsort()[::-1], 'item']
    sns.factorplot(x="item", y="consumption", data=items, kind="bar", order=item_order)
    
    