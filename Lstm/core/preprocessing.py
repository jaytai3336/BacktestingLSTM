import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataLoader:
    """
    A class to load and preprocess time series data for LSTM or sequence models.

    Args:
        filename (str): Path to the CSV file.
        train_test_split (float): Proportion of data to use for training (e.g., 0.8).
        cols (list): List of column names to use. First column is assumed to be the target.
    """
    def __init__(self, filename, train_test_split=0.8, cols=None):
        df = pd.read_csv(filename, low_memory=False)
        assert all(c in df.columns for c in cols), "Some columns in `cols` not found in dataset"

        self.cols = cols
        self.df = df[cols]
        split = int(len(df) * train_test_split)
        self.train_data = self.df.values[:split]
        self.test_data = self.df.values[split:]
        self.seq_len = None

        self.scaler_all = MinMaxScaler().fit(self.train_data)
        self.scaler_target = MinMaxScaler().fit(self.train_data[:, [0]])  # assume first col is target

    def _normalize_window(self, window, single_window=False):
        if single_window:
            return self.scaler_all.transform(window)
        else:
            n_windows, seq_len, n_features = window.shape
            reshaped = window.reshape(-1, n_features)
            scaled = self.scaler_all.transform(reshaped)
            return scaled.reshape(n_windows, seq_len, n_features)

    def _create_sequence(self, data, seq_len, normalise=True):
        X, y = [], []
        for i in range(len(data) - seq_len):
            window = data[i:i + seq_len]
            if normalise:
                window = self.scaler_all.transform(window)
            x = window[:-1]
            y_val = data[i + seq_len - 1, 0]  # target: first column
            if normalise:
                y_val = self.scaler_target.transform([[y_val]])[0][0]
            X.append(x)
            y.append(y_val)
        return np.array(X), np.array(y)

    def get_train_data(self, seq_len, normalise=True):
        self.seq_len = seq_len
        return self._create_sequence(self.train_data, seq_len, normalise)

    def get_test_data(self, seq_len, normalise=True):
        self.seq_len = seq_len
        return self._create_sequence(self.test_data, seq_len, normalise)

    def generate_train_batch(self, seq_len, batch_size, normalise=True):
        """
        Generator that yields training batches of shape (batch_size, seq_len-1, n_features)
        and targets of shape (batch_size,).
        """
        i = 0
        while True:
            x_batch, y_batch = [], []
            for _ in range(batch_size):
                if i >= len(self.train_data) - seq_len:
                    i = 0
                window = self.train_data[i:i+seq_len]
                if normalise:
                    window = self.scaler_all.transform(window)
                x = window[:-1]
                y_val = self.train_data[i + seq_len - 1, 0]
                if normalise:
                    y_val = self.scaler_target.transform([[y_val]])[0][0]
                x_batch.append(x)
                y_batch.append(y_val)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def inverse_transform_target(self, data):
        """
        Inverse transform target data from normalized to original scale.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.scaler_target.inverse_transform(data).flatten()
