# Solar simulator v0

## Why
I have a PV installation with a 5.5 kWh battery, currently it is charged with the sun and discharged as soon as the 
load is higher than the production. I would like to use the battery to store energy when the price is low and discharge
it when the price is high. This way (in theory) I can save money and use the battery more efficiently.
With the simulation I can test different strategies and see which one is the best or consider buying a bigger battery.

## How
I have collected historical data for the PV production, the load and the SoC of the battery. Due to file sizes being
too large, I have decimated the data to 1 hour intervals.

### The idea is:

1. Calculate the baseline cost for a year
2. Calculate the cost for a year with different strategies
3. Implement a driver for home assistant that executes the best strategy

### Things to consider

- We can't predict the future, so we can only use historical data
- Energy costs depend on ToD (time of day) and DoW (day of week), weekends are flat rate.
- The battery has a maximum charge and discharge rate, and a maximum capacity.
- The battery has some unusable capacity, so we can't discharge it below 10%

## Sources
- [pvlib python](https://pvlib-python.readthedocs.io/en/stable/)
- 