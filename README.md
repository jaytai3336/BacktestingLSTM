# Forecasting S&P 500 Futures: LSTM vs ARIMA-GARCH on 1-Minute Bars

A comparison of statistical (ARIMA-GARCH) and deep learning (LSTM) approaches to
short-horizon price forecasting on S&P 500 futures, using 1-minute intraday bars.
The project also tests whether adding sentiment scores and cross-model signals
(feeding ARIMA-GARCH forecasts into the LSTM) improves on either method alone.

**Headline finding:** on this data and horizon, neither model beats a coin flip on
directional accuracy. That result is treated as a finding, not a failure — see
[Results](#results) below.

## Motivation

Inspired by ["Stock Price Prediction Based on ARIMA-GARCH and LSTM"](https://onlinelibrary.wiley.com/doi/10.1155/2021/9942410),
this project asks: at 1-minute resolution, how much of S&P 500 futures price
movement is actually predictable from price history, technical indicators, and
news sentiment? ARIMA-GARCH and LSTM represent two different bets on where that
predictability lives — linear time-series structure vs. nonlinear pattern learning.

## Data

- **Instrument:** S&P 500 futures (ES), 1-minute bars
- **Features used across model variants:** OHLCV, MACD (+ signal, histogram), RSI(10),
  Bollinger Band std dev, sentiment score (from a separate [sentiment analysis
  pipeline](https://github.com/jaytai3336/Sentimental-Analysis)), and in later variants,
  ARIMA-GARCH's own forecast fed back in as a feature
- **Split:** 80/20 train/test, evaluated with walk-forward validation to approximate
  real-world deployment (no lookahead)

## Models

| Model | Inputs | Status |
|---|---|---|
| ARIMA-GARCH | Close price only | ✅ Fit and evaluated |
| LSTM v1 | Close + technical indicators (10 features) | 🚧 Architecture defined ([`config.json`](Lstm/config.json)), not yet trained/saved |
| LSTM v2 | Technical indicators + sentiment score + ARIMA-GARCH forecast (13 features) | ✅ Fit and evaluated |
| LSTM v3 | + wavelet-transformed price series | 🚧 Planned, not yet implemented |
| LSTM v4 | + SSA (singular spectrum analysis) transformed price series | 🚧 Planned, not yet implemented |

Both LSTM variants use a stacked LSTM architecture (128 → 32/64 units) with dropout
regularization, trained with Adam and early stopping on validation MSE. Full configs
are in [`Lstm/config.json`](Lstm/config.json) and [`Lstm/config2.json`](Lstm/config2.json).

## Results

### ARIMA-GARCH

Residual diagnostics on the fitted model show the ACF/PACF of residuals are
essentially flat past lag 1 — the model captured the autocorrelation structure in
the series well, and residuals are approximately normal near the center (heavier
tails at the extremes, visible in the QQ plot).

![ARIMA-GARCH residual diagnostics](Arima-Garch/results/Arima_Garch%20verification.png)

But capturing structure in *returns* is not the same as predicting *direction*.
On held-out forecasts:

| Metric | Value |
|---|---|
| RMSE (returns) | 0.00083 |
| MAE (returns) | 0.00065 |
| Directional accuracy | **44.7%** |

### LSTM v2 (technical indicators + sentiment + ARIMA-GARCH forecast)

| Metric | Value |
|---|---|
| RMSE (price level) | 11.12 points |
| MAPE | 0.18% |
| Directional accuracy | **38.2%** |

The low MAPE is misleading on its own — at this horizon the index barely moves
between bars, so predicting "no change" gets you most of the way to a low error
without any real skill. Directional accuracy is the metric that matters here, and
it's below chance.

### Reading these results

Both models are worse than a coin flip at predicting the direction of the next
1-minute move. That's a real result, not a bug: at 1-minute resolution on one of
the most liquid, heavily-traded instruments in the world, price changes are close
to indistinguishable from noise — which is exactly what efficient-market theory
predicts. The ARIMA-GARCH residual diagnostics confirm the model isn't *missing*
linear structure; there's very little linear structure left to find once you
account for the series' own autocorrelation.

The sentiment-augmented and ARIMA-GARCH-fused model (v2) is the only LSTM variant
trained and evaluated so far. The plain technical-indicators-only LSTM (v1) was
configured but never trained, so it's not yet possible to say whether sentiment
and the ARIMA-GARCH signal actually helped or hurt — that comparison, plus the
wavelet/SSA variants (v3, v4), is the natural next step (see
[Next steps](#next-steps)).

## Project structure

```
├── Arima-Garch/
│   ├── Arima_Garch_Training.py      # ARIMA-GARCH fit + forecast
│   ├── Arimax_EGarch_Training.py    # ARIMAX-EGARCH variant
│   └── results/                     # Forecast output, residual diagnostics
├── Lstm/
│   ├── Lstm_training_sentiments.py  # LSTM v2: technical + sentiment + ARIMA-GARCH features
│   ├── training/
│   │   └── Lstm_training_technicals.py  # LSTM v1: technical indicators only
│   ├── config.json                  # LSTM v1 config
│   ├── config2.json                 # LSTM v2 config
│   └── saved_models/                # Trained model checkpoints + training histories
├── data/
│   ├── raw/                         # Raw intraday data
│   └── processed/                   # Feature-engineered datasets
└── notebooks/
    ├── EDA.ipynb                    # Exploratory analysis
    └── Comparison.ipynb             # Model comparison (in progress)
```

## How to run

```bash
pip install -r requirements.txt

# Fit ARIMA-GARCH and generate forecasts
python Arima-Garch/Arima_Garch_Training.py

# Train the technical-indicators LSTM
python Lstm/training/Lstm_training_technicals.py

# Train the sentiment + ARIMA-GARCH fused LSTM
python Lstm/Lstm_training_sentiments.py
```

## Next steps

- [ ] Train the plain technical-indicators LSTM (v1) and benchmark it against v2 on the same holdout window — needed to isolate whether sentiment/ARIMA-GARCH fusion actually helps
- [ ] Implement wavelet-transformed (v3) and SSA-transformed (v4) variants
- [ ] Extend evaluation across short/medium/long horizons (3 days, 1 month, 3 months) as originally planned
- [ ] Test on a calmer instrument or lower-frequency bars to see whether directional accuracy improves when the signal-to-noise ratio is more favorable

## Notes

This started as an exploration of static vs. dynamic forecasting (using a model's
own outputs to forecast further into the future) — the current setup focuses on
static, walk-forward forecasting to approximate realistic backtesting conditions.
