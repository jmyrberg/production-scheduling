# Item production scheduling
In this repository, mathematical optimization is used for solving an item production scheduling problem.

### Problem description
When manufacturing items on a machine, the machine may consume varying amounts of electricity per hour, depending on the item that is being produced. The idea is to utilize this variation when scheduling the production of items. For example, it may make sense to produce high-consuming items to hours where the electricity price is expected to be low.

Given a realized production schedule of a machine in the past, we assess whether cost savings could've been achieved, if the schedule had been optimized.

The following three-step approach is used for solving the problem:
1. Generate electricity forecast for the future
2. Solve the problem as an MILP (mixed integer linear programming)
3. Compare results with realized schedule

### Code & files
#### Input data
* **items.csv**: Electricity consumption for each item per 4-hour production block
* **prices.csv**: Realized hourly electricity prices from 1.1.2014 -
* **schedule.csv**: Realized production schedule for time period 29.9.2016 - 11.10.2016
#### Python files
* **data_exploration.py**:  Plotting input data and results
* **data_utils.py**: Data loading and preprocessing
* **forecast.py**: Electricity price forecast model
* **optimization.py**: Production scheduling optimization model
* **model.py**: Running the entire model and plotting results

### Results
The following results can be obtained by simply running **model.py**.

#### Realized schedule
(figures/realized.png)

#### Theoretical optimum model (100% electricity price forecast accuracy)
(figures/theoretical.png)

#### MedianForecaster model
The electricity price forecast:
(figures/medianforecaster_electricity.png)

Results:
(figures/medianforecaster.png)

The results show that cost savings can be achieved even by using optimization, and a simple model for electricity price forecast.
