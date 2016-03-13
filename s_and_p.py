
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pylab 
import statsmodels.api as sm 
import seaborn as sb 
sb.set_style ('darkgrid')


#Note: I have commented multiple lines, uncommenting these lines will produce a plot of
# the data in each stage.




path = os.getcwd() + '/q_table.csv'
stock_data = pd.read_csv (path)
stock_data ['Date'] = pd.to_datetime(stock_data['Date'])
stock_data = stock_data.sort_values(by='Date', ascending=True)
stock_data = stock_data.set_index('Date') 
#stock_data['Close'].plot(figsize=(16, 12))
#pylab.show ()
stock_data['First Difference'] = stock_data['Close'] - stock_data['Close'].shift ()
#stock_data['First Difference'].plot (figsize= (10,10))
#pylab.show ()

#Applying log-Transform from numpy 'np'
stock_data ['Natural Log'] = stock_data['Close'].apply(lambda x: np.log (x)) 
#stock_data['Natural Log'].plot (figsize= (16,12))
#pylab.show ()

stock_data['Original Variance'] = pd.rolling_var(stock_data['Close'], 30, min_periods=None, freq=None, center=True)  
stock_data['Log Variance'] = pd.rolling_var(stock_data['Natural Log'], 30, min_periods=None, freq=None, center=True)

#fig, ax = plt.subplots(2, 1, figsize=(13, 12))  
#stock_data['Original Variance'].plot(ax=ax[0], title='Original Variance')  
#stock_data['Log Variance'].plot(ax=ax[1], title='Log Variance')  
#fig.tight_layout()  
#pylab.show ()

stock_data['Logged First Difference'] = stock_data['Natural Log'] - stock_data['Natural Log'].shift(1)  
#stock_data['Logged First Difference'].plot(figsize=(16, 12))  
#pylab.show ()


#Trying out different Lagging possibilities.
#stock_data['Lag 1'] = stock_data ['Logged First Difference'].shift (1)
#stock_data['Lag 5'] = stock_data ['Logged First Difference'].shift (5)
#stock_data['Lag 10'] = stock_data ['Logged First Difference'].shift (10)
stock_data['Lag 20'] = stock_data ['Logged First Difference'].shift (20)


# Checking out a scatter plot , probably we can try out different lags and check data
#sb.jointplot('Logged First Difference','Lag 20',stock_data, kind ='reg', size = 10)
#pylab.show ()

# Probably we can use stat models and check out the lagged data for all and see
#if any correlation exits

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

#acf is auto correlation fucntion and pacf is partial acf (works only for 1 d array)
#iloc is integer location, check pandas

lag_corr = acf (stock_data ['Logged First Difference'].iloc [1:])
lag_partial_corr = pacf (stock_data ['Logged First Difference'].iloc [1:])

#fig, ax = plt.subplots (figsize = (16,12))
#ax.plot (lag_corr)
#pylab.show ()

# To extract trends and seasonal patterns for TS analysis

from statsmodels.tsa.seasonal import seasonal_decompose

#set the frequency value right for monthly set freq = 30
decomposition = seasonal_decompose(stock_data['Natural Log'], model='additive', freq=30)  
#fig = decomposition.plot()  
#pylab.show ()

#lets fit some ARIMA, keep indicator as 1 and rest as zero ie (p,q,r) = (1,0,0)
#the snippet below does it for undifferenced series

#model = sm.tsa.ARIMA (stock_data ['Natural Log'].iloc[1:], order = (1,0,0))
#result = model.fit (disp = -1)
#stock_data ['Forecast'] = result.fittedvalues
#stock_data [['Natural Log', 'Forecast']].plot (figsize = (16,12))
#pylab.show ()

#trying an exponential smoothing model
model = sm.tsa.ARIMA(stock_data['Logged First Difference'].iloc[1:], order=(0, 0, 1))  
results = model.fit(disp=-1)  
stock_data['Forecast'] = results.fittedvalues  
stock_data[['Logged First Difference', 'Forecast']].plot(figsize=(16, 12)) 
pylab.show ()

# Looking at exponenetial smoothing model, doesnt help much either,  but probably with actual data
#from IBM we can probably predict the method better suited for that data.







