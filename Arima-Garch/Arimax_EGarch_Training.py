import pandas as pd
import numpy as np
from core.model.ArimaX_EGarch import ARIMAX_EGARCH_Model

# -------------------------------
# 🔽 Example usage (customize here)
# -------------------------------
if __name__ == "__main__":
    # Load your CSV or DataFrame
    df = pd.read_csv("data/processed/spy_sentiment_processed_mini.csv")  # Replace with actual path
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)

    # Required columns
    df['log_return'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
    price_col = 'log_return'
    exog_cols = ['sentiment_score', 'cluster']

    # Drop NaNs
    df = df[[price_col] + exog_cols].dropna()

    model = ARIMAX_EGARCH_Model(arima_order=(1,1,1), garch_order=(1,1))
    results = model.fit_walk_forward(df[price_col], df[exog_cols])
    model.evaluate_and_plot()
