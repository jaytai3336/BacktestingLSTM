# Dataloader
import pandas as pd
import numpy as np

class Dataloader():
    def __init__(self, data, datetime_col = 'DATE', close_col = 'CLOSE'):
        self.raw = data.copy()
        self.datetime_col = datetime_col
        self.price_col = close_col
        self.processed = None

    def preprocess(self, exog_cols=None, scale_exog=False, fillna_method='ffill'):
        df = self.raw.copy()

        # 1. Parse datetime and set index
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df = df.set_index(self.datetime_col)
        df = df.sort_index()

        # 2. Handle missing values
        df = df.fillna(method=fillna_method)

        # 3. Compute log returns
        df['log_return'] = np.log(df[self.price_col] / df[self.price_col].shift(1))
        df = df.dropna()

        # 4. Exogenous variable handling (multi-column)
        if exog_cols:
            missing_cols = [col for col in exog_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing exog columns in data: {missing_cols}")

            exog_df = df[exog_cols].copy()
            if scale_exog:
                exog_df = (exog_df - exog_df.mean()) / exog_df.std()

            self.exog = exog_df
        else:
            self.exog = None

        self.processed = df
        self.returns = df['log_return']
        return df

    def get_data(self):
        if self.processed is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        return self.returns, self.exog