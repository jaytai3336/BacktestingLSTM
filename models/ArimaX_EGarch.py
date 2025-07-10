import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback
import warnings
warnings.filterwarnings("ignore")

class Arima_Garch:
    def __init__(self):
        self.arima_model = None
        self.residuals = None
        self.garch_model = None
        self.forecast_results = None

    def verify(self, log_returns, tweets, arima_order=(1,1,1), garch_order=(1,1)):
        print("\n🔧 Fitting ARIMAX model...")
        self.arima_model = SARIMAX(log_returns, exog=tweets, order=arima_order).fit()
        self.residuals = self.arima_model.resid.dropna()
        print(self.arima_model.summary())

        print("\n📊 Residual Diagnostics:")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # ACF & PACF
        plot_acf(self.residuals, ax=axes[0, 0], lags=40)
        axes[0, 0].set_title("ACF of ARIMAX Residuals")
        plot_pacf(self.residuals, ax=axes[0, 1], lags=40)
        axes[0, 1].set_title("PACF of ARIMAX Residuals")

        # Histogram
        sns.histplot(self.residuals, bins=50, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("Histogram of ARIMAX Residuals")

        # QQ Plot
        stats.probplot(self.residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("QQ Plot of ARIMAX Residuals")
        plt.tight_layout()
        plt.show()

        # Ljung-Box test (autocorrelation)
        lb_test = acorr_ljungbox(self.residuals, lags=[10], return_df=True)
        print("🧪 Ljung-Box test (lag 10):")
        print(lb_test)

        # ARCH LM test (heteroskedasticity)
        arch_test = het_arch(self.residuals)
        print("\n🧪 ARCH LM test:")
        print(f"LM Stat: {arch_test[0]:.4f}, p-value: {arch_test[1]:.4f}")

        print("\n📈 Fitting EGARCH model...")
        self.garch_model = arch_model(self.residuals, vol='EGARCH',
                                    p=garch_order[0], q=garch_order[1])
        self.garch_model = self.garch_model.fit(disp='off')
        print(self.garch_model.summary())

        # Volatility Plot
        self.garch_model.conditional_volatility.plot(figsize=(10, 4), title='Estimated Conditional Volatility')
        plt.show()

    def walk_forward_forecast(self, log_returns, tweets, train_size=0.8, arima_order=(1,1,1), garch_order=(1,1)):
        n = len(log_returns)
        train_len = int(train_size * n)

        forecasts_mean = []
        forecasts_vol = []
        actual = []

        for i in range(train_len, n-1):
            train = log_returns[:i]

            # Skip problematic inputs
            if len(train) < max(arima_order) + 5:
                continue
            if train.isnull().any() or np.isinf(train).any():
                continue

            try:
                # Fit ARIMA
                arima_model = SARIMAX(train, tweets, order=arima_order).fit()
                forecast_mean = arima_model.forecast(steps=1).iloc[0]
                residuals = arima_model.resid

                # Fit GARCH
                garch = arch_model(residuals, vol='EGARCH', p=garch_order[0], q=garch_order[1])
                garch_model = garch.fit(disp='off')

                # Forecast variance
                forecast = garch_model.forecast(horizon=1)
                var = forecast.variance.values
                if np.isnan(var[-1, 0]) or np.isinf(var[-1, 0]):
                    raise ValueError("Invalid GARCH forecast variance")
                
                forecast_vol = np.sqrt(var[-1, 0])

                # Save results
                forecasts_mean.append(forecast_mean)
                forecasts_vol.append(forecast_vol)
                actual.append(log_returns[i+1])

            except Exception as e:
                print(f"Warning: Skipped iteration {i}")
                traceback.print_exc()
                continue

        # Convert to DataFrame
        self.results = pd.DataFrame({
            'Forecast_Mean': forecasts_mean,
            'Forecast_Std': forecasts_vol,
            'Actual': actual
        })

        return self.results
    
    def forecast_results(self):
        if not hasattr(self, 'results') or self.results.empty:
            print("❌ No forecast results available. Run walk_forward_forecast() first.")
            return

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(self.results['Actual'].values, label='Actual')
        plt.plot(self.results['Forecast_Mean'].values, label='Forecast Mean')
        plt.fill_between(self.results.index,
                        self.results['Forecast_Mean'] - 1.96 * self.results['Forecast_Std'],
                        self.results['Forecast_Mean'] + 1.96 * self.results['Forecast_Std'],
                        color='lightgray', label='95% CI')
        plt.title("ARIMAX-EGARCH 1-step Walk-Forward Forecast")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Metrics
        actual = self.results['Actual']
        predicted = self.results['Forecast_Mean']
        std = self.results['Forecast_Std']

        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        # Confidence interval coverage
        lower = predicted - 1.96 * std
        upper = predicted + 1.96 * std
        inside_ci = ((actual >= lower) & (actual <= upper)).mean()

        # Print metrics
        print("📈 Forecast Evaluation Metrics:")
        print(f"🔹 Mean Squared Error (MSE): {mse:.6f}")
        print(f"🔹 Mean Absolute Error (MAE): {mae:.6f}")
        print(f"🔹 R² Score: {r2:.4f}")
        print(f"🔹 95% CI Coverage: {inside_ci * 100:.2f}% of actual values within interval")









