import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime
from core.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['date'] = sorted(self.df['date'])
        self.scaler = StandardScaler()
        self.training_data_len = 0
        self.scaled_data = None

    def preprocess_data(self):
        stock_close = self.df.filter(["close"])
        dataset = stock_close.values  # converting to numpy array
        self.training_data_len = int(np.ceil(len(dataset) * .95))
        self.scaled_data = self.scaler.fit_transform(dataset)

    def get_training_data(self):
        return self.scaled_data[0:self.training_data_len, :]