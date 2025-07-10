import pandas as pd
import numpy as np

import sys
import os
folder_path = os.path.abspath('models')
sys.path.append(folder_path)

from Arima_Garch import Arima_Garch

import warnings
warnings.filterwarnings("ignore")

# Data loading and preprocessing
df = pd.read_excel('data/raw/SnP futures intraday.xlsx')
df = df.rename(columns={'Time at end of bar': 'DATE', 'OPEN': 'OPEN', 'HIGH': 'HIGH',
                       'LOW': 'LOW', 'CLOSE': 'CLOSE', 'VOLUME': 'VOLUME'})
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')
df.index = df['DATE']

log_returns = np.log(df['CLOSE']).diff().dropna()

model = Arima_Garch()
# model.verify(log_returns)

results = model.walk_forward_forecast(log_returns)
model.plot_forecast()

df = df.join(results, how='right')

df.to_excel('findings/Arima_Garch/Arima_Garch_Forecast.xlsx')