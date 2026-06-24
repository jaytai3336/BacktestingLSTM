import pandas as pd
import numpy as np
import argparse
from core.model.ArimaX_EGarch import ARIMAX_EGARCH_Model

# -------------------------------
# 🔽 Example usage (customize here)
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit ARIMAX-EGARCH with sentiment/cluster exogenous variables."
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help=(
            "Path to a CSV with columns: DATE, CLOSE, sentiment_score, cluster. "
            "This script needs a sentiment-merged dataset that isn't included in this repo "
            "— see the README's data note."
        )
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
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
