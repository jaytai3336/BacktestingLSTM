import pandas as pd
import numpy as np

import sys
import os
import argparse
folder_path = os.path.abspath('models')
sys.path.append(folder_path)

from Arima_Garch import Arima_Garch

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Fit ARIMA-GARCH on intraday OHLCV data.")
parser.add_argument(
    '--data-path',
    type=str,
    default='data/raw/SnP futures intraday.xlsx',
    help="Path to an Excel file with intraday OHLCV data and a 'Time at end of bar' column."
)
args = parser.parse_args()

# Data loading and preprocessing
df = pd.read_excel(args.data_path)
df = df
df = df.rename(columns={'Time at end of bar': 'DATE', 'OPEN': 'OPEN', 'HIGH': 'HIGH',
                       'LOW': 'LOW', 'CLOSE': 'CLOSE', 'VOLUME': 'VOLUME'})
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')
df.set_index('DATE', inplace=True)

log_returns = np.log(df['CLOSE']).diff().dropna()

model = Arima_Garch()

# model.verify(log_returns)

try:
  model.walk_forward_forecast(log_returns)
  model.plot_and_evaluate_forecast()

finally:
  df_merged = df.merge(model.get_results(), left_index = True, right_index=True, how='left')
  os.makedirs('findings/Arima_Garch', exist_ok=True)
  df_merged.to_csv('findings/Arima_Garch/Arima_Garch_forecast.csv')