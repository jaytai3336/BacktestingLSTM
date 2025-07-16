import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import seaborn as sns
import scipy.stats as stats
import warnings
import contextlib
import io
import traceback

warnings.filterwarnings("ignore")

class ARIMAX_EGARCH_Model:
    def __init__(self, arima_order=(1,1,1), garch_order=(1,1), train_size=0.8):
        self.arima_order = arima_order
        self.garch_order = garch_order
        self.train_size = train_size
        self.results = None

    def fit_walk_forward(self, log_returns, exog_df):
        n = len(log_returns)
        train_len = int(self.train_size * n)

        forecasts_mean = []
        forecasts_vol = []
        actual = []

        for i in range(train_len, n - 1):
            y_train = log_returns[:i]
            exog_train = exog_df.iloc[:i]
            exog_next = exog_df.iloc[i:i+1]

            if y_train.isnull().any() or np.isinf(y_train).any():
                continue

            try:
                # Fit ARIMAX silently
                with contextlib.redirect_stdout(io.StringIO()):
                    arima = SARIMAX(y_train, exog=exog_train, order=self.arima_order)
                    arima_result = arima.fit(disp=False)

                forecast_mean = arima_result.forecast(steps=1, exog=exog_next).iloc[0]
                residuals = arima_result.resid.dropna()

                # Fit EGARCH silently
                with contextlib.redirect_stdout(io.StringIO()):
                    garch = arch_model(residuals, vol='GARCH', p=self.garch_order[0], q=self.garch_order[1])
                    garch_result = garch.fit(disp='off')

                forecast_var = garch_result.forecast(horizon=1).variance.values[-1, 0]
                if np.isnan(forecast_var) or np.isinf(forecast_var):
                    raise ValueError("Invalid GARCH variance")

                forecasts_mean.append(forecast_mean)
                forecasts_vol.append(np.sqrt(forecast_var))
                actual.append(log_returns.iloc[i+1])

            except Exception:
                print(f"⚠️ Skipped iteration {i}")
                traceback.print_exc()
                continue

        self.results = pd.DataFrame({
            'Forecast_Mean': forecasts_mean,
            'Forecast_Std': forecasts_vol,
            'Actual': actual
        })

        return self.results

    def evaluate_and_plot(self):
        if self.results is None or self.results.empty:
            print("❌ No forecast results available.")
            return

        actual = self.results['Actual']
        predicted = self.results['Forecast_Mean']
        std = self.results['Forecast_Std']

        # Metrics
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        ci_coverage = ((actual >= predicted - 1.96*std) & (actual <= predicted + 1.96*std)).mean()

        print("📈 Forecast Evaluation:")
        print(f"🔹 MSE: {mse:.6f}")
        print(f"🔹 MAE: {mae:.6f}")
        print(f"🔹 R²: {r2:.4f}")
        print(f"🔹 95% CI Coverage: {ci_coverage * 100:.2f}%")

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(actual.values, label="Actual")
        plt.plot(predicted.values, label="Forecast Mean")
        plt.fill_between(self.results.index,
                         predicted - 1.96 * std,
                         predicted + 1.96 * std,
                         color='gray', alpha=0.3, label='95% CI')
        plt.legend()
        plt.title("ARIMAX-EGARCH Walk-Forward Forecast")
        plt.tight_layout()
        plt.show()