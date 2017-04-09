'''
Created on 14.1.2017

@author: Jesse Myrberg / jesse.myrberg@gmail.com
'''
import pandas as pd
import numpy as np
from data_utils import itemnames
from collections import defaultdict
from pulp import LpProblem,LpMinimize,LpVariable,LpStatus,value
    

def create_optimization_problem(itemblocks_to_produce,blocks_available,
                                forecasted_block_prices,item_consumptions,
                                name="Production scheduling"):
    '''
    PROBLEM DESCRIPTION
    This function formulates the following optimization problem:
    Schedule blocks <itemblocks_to_produce> of different paper item in four-hour-blocks <blocks_available>,
    so that the cost of producing them is minimized. Different items have different electricity 
    consumption <item_consumptions> per block, and also the price of electricity <forecasted_block_prices> 
    depends on the block.
    
    OBJECTIVE
    minimize c * X
    
    VARIABLES
    o x_ij in X is the decision variable, whether to produce item j at block i (in [0,1])
    
    PARAMETERS
    o Block i: 0 <= i <= T
    o Grade j: 0 <= j <= M=21
    o m_j is the electricity consumption per block for item j (predicted)
    o p_i is the average price of electricity at block i
    o c_ij = p_i * m_j is the cost of running block i with item j.
    o S_j is the number of blocks of item j that must be produced within the given timeframe
    
    CONSTRAINTS
    o Decision variable is binary: x_ij is binary for all i,j
    o All itemblocks should be produced: x_1j + x_2j + x_3j + ... + x_(T-1)j + x_Tj = S_j, for all j
    o Maximum of one item can be produced at a time: x_i1 + x_i2 + x_i3 + ... + x_i(M-1) + x_iM <= 1
    '''
    
    # INITIALIZATION
    # Calculate cost coefficients
    p = forecasted_block_prices
    m = item_consumptions
    c = {}
    for kp,vp in p.items():
        for km,vm in m.items():
            c[(kp,km)] = vp*vm
    
    # Problem variable
    prob = LpProblem(name,LpMinimize)
    
    # Decision variables
    all_blocks = blocks_available
    all_items = [g for g in itemnames() if g in list(itemblocks_to_produce.keys())]
    all_combinations = []
    for i in all_blocks:
        for j in all_items:
            all_combinations.append((i,j))
    X = LpVariable.dicts(name='decision', indexs=all_combinations, lowBound=0, upBound=1, cat='Binary')
    
    # OBJECTIVE
    prob += sum([c[comb] * X[comb] for comb in all_combinations])
    
    
    # CONSTRAINTS
    # All itemblocks should be produced
    S = itemblocks_to_produce
    for j in all_items:
        combs = [(i,j) for i in all_blocks]
        prob += sum([X[comb] for comb in combs]) == S[j], "All %s should be produced" % str(j)
    
    # Maximum of one item can be produced at a time
    for i in all_blocks:
        combs = [(i,j) for j in all_items]
        prob += sum([X[comb] for comb in combs]) <= 1, "Maximum of one item can be produced at block %s at a time" % str(i)
        
        
    # Save problem
    prob.writeLP("./lp/"+name+".lp")
    
    return(prob)

def solve_problem(prob,actual_block_prices,item_consumptions,block_order):
    '''Solve optimization problem defined in function 'create_optimization_problem'.
    
    This function returns a dataframe with the results of the problem.'''
    
    # Solve
    prob.solve()
    
    # Print and save result
    print("Status:", LpStatus[prob.status])
    solution = defaultdict(dict)
    for v in prob.variables():
        name = v.name
        val = int(v.varValue)
        if val > 0:
            print(name + " = " + str(val))
        ind = name[9:]
        ind = ind.split(",")[0].replace("_","-") + "," + ind.split(",")[1][1:]
        ind = eval(ind)
        blockid, item = ind[0],ind[1]
        solution[blockid][item] = val
    
    # Create dataframe
    df = pd.DataFrame(solution).T
    df = df[itemnames()]
    prices = pd.DataFrame.from_dict(actual_block_prices, 'index')
    prices.columns = ['price']
    df = df.merge(prices, how='left', left_index=True, right_index=True)
    df = df.reindex(block_order)
    df['consumption'] = df[itemnames()].apply(lambda x: x * pd.DataFrame.from_dict(item_consumptions,'index').T[itemnames()].as_matrix().T.flatten(), axis=1).sum(1)
    df['cost'] = df.consumption * df.price
    df['cumulative_cost'] = df.cost.cumsum()
    df.replace(0,np.nan,inplace=True)
    df.index.name = 'blockid'
    
    print("Forecasted total cost = ", value(prob.objective))
    print("Actual total cost = ", df.cumulative_cost.max())
    return(df)
    