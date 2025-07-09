
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

df_close = df['CLOSE'].copy()
log_returns = np.log(df_close / df_close.shift(1)).dropna()

arima_model = ARIMA(log_returns, order=(1, 1, 1)).fit()
residuals = arima_model.resid

print(arima_model.summary())

garch_model = arch_model(residuals, vol='GARCH', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

print(garch_fit.summary())

plt.figure(figsize=(10, 4))
plt.plot(garch_fit.conditional_volatility, label='Conditional Volatility')
plt.title(f'GARCH Conditional Volatility')
plt.legend()
plt.tight_layout()
plt.show()

'''
                               SARIMAX Results
==============================================================================
Dep. Variable:                  CLOSE   No. Observations:               185879
Model:                 ARIMA(1, 1, 1)   Log Likelihood             1168632.128
Date:                Wed, 09 Jul 2025   AIC                       -2337258.256
Time:                        18:04:01   BIC                       -2337227.857
Sample:                             0   HQIC                      -2337249.280
                             - 185879
Covariance Type:                  opg
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.0366      0.000    -88.341      0.000      -0.037      -0.036
ma.L1         -0.7263      0.000  -2223.632      0.000      -0.727      -0.726
sigma2      1.979e-07   4.14e-11   4783.754      0.000    1.98e-07    1.98e-07
===================================================================================
Ljung-Box (L1) (Q):                2567.44   Jarque-Bera (JB):        2199268084.30
Prob(Q):                              0.00   Prob(JB):                         0.00
Heteroskedasticity (H):               1.07   Skew:                             1.84
Prob(H) (two-sided):                  0.00   Kurtosis:                       535.87
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
C:\Users\Jay Tai\Documents\BacktestingLSTM\.venv\Lib\site-packages\arch\univariate\base.py:768: ConvergenceWarning: The optimizer returned code 4. The message is:
Inequality constraints incompatible
See scipy.optimize.fmin_slsqp for code meaning.

  warnings.warn(
                     Constant Mean - GARCH Model Results
==============================================================================
Dep. Variable:                   None   R-squared:                       0.000
Mean Model:             Constant Mean   Adj. R-squared:                  0.000
Vol Model:                      GARCH   Log-Likelihood:            1.26456e+06
Distribution:                  Normal   AIC:                      -2.52911e+06
Method:            Maximum Likelihood   BIC:                      -2.52907e+06
                                        No. Observations:               185879
Date:                Wed, Jul 09 2025   Df Residuals:                   185878
Time:                        18:04:22   Df Model:                            1
                                  Mean Model
=============================================================================
                 coef    std err          t      P>|t|       95.0% Conf. Int.
-----------------------------------------------------------------------------
mu         1.5260e-09  2.012e-06  7.586e-04      0.999 [-3.941e-06,3.944e-06]
                              Volatility Model
============================================================================
                 coef    std err          t      P>|t|      95.0% Conf. Int.
----------------------------------------------------------------------------
omega      4.0510e-09  6.341e-11     63.884      0.000 [3.927e-09,4.175e-09]
alpha[1]       0.2000  5.036e-02      3.971  7.143e-05     [  0.101,  0.299]
beta[1]        0.7800  5.807e-02     13.431  3.961e-41     [  0.666,  0.894]
============================================================================

Covariance estimator: robust
WARNING: The optimizer did not indicate successful convergence. The message was Inequality constraints incompatible.
See convergence_flag.
'''