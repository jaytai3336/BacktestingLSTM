import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataLoader():
    def __init__(self, filename, train_test_split, cols):
        df = pd.read_csv(filename, low_memory=False) # Change when dealing with other datasets
        assert all(c in df.columns for c in cols), "Some columns in `cols` not found in dataset"
        split = int(len(df) * train_test_split)
        self.df_train = df.get(cols).values[:split]
        self.df_test = df.get(cols).values[split:]
        self.len_train = len(self.df_train)
        self.len_test = len(self.df_test)
        self.len_train_window = None
        self.scaler_all = MinMaxScaler()
        self.scaler_all.fit(self.df_train)
        self.scaler_target = MinMaxScaler()
        self.scaler_target.fit(self.df_train[:, [0]])

    def get_train_data(self, seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(self.len_train-seq_len):
            x,y = self.next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)
    
    def get_test_data(self, seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(self.len_test-seq_len):
            x,y = self.next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)
    
    def generate_train_batch(self, seq_len, batch_size, normalise):
        i = 0
        while True:  # loop forever
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    i = 0  # restart from beginning
                x, y = self.next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)


    # def next_window(self, i, seq_len, normalise):
    #     window = self.df_train[i:i+seq_len]
    #     window = self.normalise_window(window, single_window = True)[0] if normalise else window
    #     x = window[:-1] 
    #     y = window[-1, [0]] 
    #     return x,y
    
    # def normalise_window(self, window_data, single_window=False):
    #     """
    #     Normalize window data by dividing by the first value of each feature and subtracting 1.
        
    #     Args:
    #         window_data (np.ndarray): Shape (n_windows, seq_len, n_features) or (seq_len, n_features) if single_window=True.
    #         single_window (bool): Whether the input is a single window (2D array).
        
    #     Returns:
    #         np.ndarray: Normalized data with same shape as input.
        
    #     Raises:
    #         ValueError: If input shape is invalid or contains zero in first values.
    #     """
    #     normalised_data = []
    #     window_data = [window_data] if single_window else window_data
    #     for window in window_data:
    #         normalised_window = []
    #         for col_i in range(window.shape[1]):
    #             normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
    #             normalised_window.append(normalised_col)
    #         normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
    #         normalised_data.append(normalised_window)
    #     return np.array(normalised_data)
    
    def normalise_window(self, window_data, single_window=False):
        if single_window:
            # Normalize features using all-feature scaler
            normed = self.scaler_all.transform(window_data)
            return normed
        else:
            # For batch of windows
            n_windows, seq_len, n_features = window_data.shape
            reshaped = window_data.reshape(-1, n_features)
            scaled = self.scaler_all.transform(reshaped)
            return scaled.reshape(n_windows, seq_len, n_features)

        
    def unnormalise(self, data):
        """
        Inverse transform normalized data.
        
        Args:
            data (np.ndarray): Normalized data with shape (samples, features) or (samples,)
        
        Returns:
            np.ndarray: Data transformed back to original scale.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.scaler_target.inverse_transform(data).flatten()

    def next_window(self, i, seq_len, normalise):
        window = self.df_train[i:i+seq_len]
        if normalise:
            window = self.normalise_window(window, single_window=True)
        x = window[:-1]
        y = window[-1, [0]]
        return x, y