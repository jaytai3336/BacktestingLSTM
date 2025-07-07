Explroing the usage of LSTM and stock backtesting

static vs dynamic forecasting? -> using own values to predict future values
focus on static forecasting (backtesting akin)

Inspired by https://onlinelibrary.wiley.com/doi/10.1155/2021/9942410
Stock Price Prediction Based on ARIMA-GARCH and LSTM, article paper

using a 80-20 train split, where the test will be done using walk forward validation to simulate real world conditions

Arima Garch - purely close data
LSTM v1 - Using close data + technical indicators
LSTM v1 - fitted with sentiment analysis indicators
LSTM v2 - fitted with predictions from Arima Garch
LSTM v3 - Close data wavelet transformed, Technical Indicators, Arima Garch predictions
LSTM v4 - Close data SSA transformed, Technical indicators, Arima Garch predictions

Compared between (short, medium, long)
3d, 1month, 3months -> best model used to predict full
(number of 1 min bars)
