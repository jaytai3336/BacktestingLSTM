
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
import pmdarima
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Data loading and preprocessing
df = pd.read_excel('data/SnP futures intraday.xlsx')
df = df.rename(columns={'Time at end of bar': 'DATE', 'OPEN': 'OPEN', 'HIGH': 'HIGH',
                       'LOW': 'LOW', 'CLOSE': 'CLOSE', 'VOLUME': 'VOLUME'})
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

# Split train and test
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]
train['CLOSE_boxcox'], lambda_value = boxcox(train['CLOSE'])

# determine the p,d,q values using pmdarima's auto_arima
# arima_model_fitted = pmdarima.auto_arima(df['CLOSE_boxcox'])
# p, d, q = arima_model_fitted.order
# arima_residuals = arima_model_fitted.arima_res_.resid
# print(f"ARIMA model order: p={p}, d={d}, q={q}")
# print(f"ARIMA model residuals: {arima_residuals}")
# ARIMA model order: p=0, d=1, q=1

# Fit ARIMA model
arima_model = ARIMA(train['CLOSE_boxcox'], order=(0, 1, 1)).fit()
print(arima_model.summary())
residuals = arima_model.resid
residuals = inv_boxcox(residuals, lambda_value)  # Box-Cox transform residuals

# # Plot ARIMA residuals and squared residuals
# plt.figure(figsize=(12, 6))
# plt.subplot(211)
# plt.plot(residuals, label='ARIMA Residuals')
# plt.title('ARIMA(0, 1, 1) Residuals (Box-Cox Scale)')
# plt.legend()
# plt.grid(True)
# plt.subplot(212)
# plt.plot(residuals**2, label='Squared Residuals')
# plt.title('Squared Residuals (Check for Volatility Clustering)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Test for heteroskedasticity
arch_test = het_arch(residuals.dropna())
print('ARCH-LM Test p-value:', arch_test[1])
if arch_test[1] < 0.05:
    print("Residuals show heteroskedasticity. Proceed with GARCH.")
else:
    print("No significant heteroskedasticity. GARCH may not be needed.")
    # If no heteroskedasticity, reconsider GARCH or check data
# Due to High p-value, we will not proceed with GARCH model.

# # Static vs Dynamic Forecasting
# # Dynamic Forecasting (Multi-step)
# forecast_obj = arima_model.get_forecast(steps=len(test))
# forecast_boxcox = forecast_obj.predicted_mean
# forecast_ci = forecast_obj.conf_int()

# # Inverse Transform
# forecast = inv_boxcox(forecast_boxcox, lambda_value)

# # Plotting the forecast
# plt.figure(figsize=(12, 6))
# plt.plot(test['DATE'], test['CLOSE'], label='Actual', color='blue')
# plt.plot(test['DATE'], forecast, label='Static Forecast', color='orange')
# plt.title('Static Forecast vs Actual')
# plt.xlabel('Date')
# plt.ylabel('S&P Futures Close Price')
# plt.legend()
# plt.show()
