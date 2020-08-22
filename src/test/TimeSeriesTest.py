import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


plt.style.use("fivethirtyeight")
from pylab import rcParams

rcParams['figure.figsize'] = 10, 7

df = pd.read_csv('../data/repository/grape.csv')
#df['Time'] = df['Year'].astype(str) + '-' +df['Month'].astype(str)
#df['Time'] = pd.to_datetime(df['Time'])
df['Date'] = pd.date_range('2000-01-01', periods=df.shape[0],
                           freq='H')
df.set_index('Date', inplace=True)

df.drop(['Year', 'Month'], axis=1, inplace=True)
print(df.head)


#plt.xlabel("Time")
#plt.ylabel("CO2")
#plt.title("Graph")
#plt.plot(df['CO2'])

#result = seasonal_decompose(df, model='multiplicative')
#result.plot()


model = ExponentialSmoothing(df['CO2'], trend='add', seasonal='add')
hw_model = model.fit(optimized=True)
pred = hw_model.predict(df.shape[0], df.shape[0] + 10)
plt.plot(df['CO2'], label='Train')
plt.plot(pred.index, pred, label = 'Test')
plt.show()