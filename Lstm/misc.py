import argparse
import pandas as pd
from core.model import Model
from core.preprocessing import SequenceGenerator

parser = argparse.ArgumentParser(
    description="Run a saved LSTM model over a full dataset and save predictions alongside it."
)
parser.add_argument(
    '--data-path',
    type=str,
    required=True,
    help=(
        "Path to a CSV with columns: CLOSE, VOLUME, OPEN, HIGH, LOW, MACD, "
        "MACD_Signal, MACD_Hist, RSI_10, BB_Std, sentiment_score, cluster. "
        "This dataset isn't included in this repo — see the README's data note."
    )
)
parser.add_argument(
    '--model-path',
    type=str,
    default='Lstm/saved_models/0715_140911_mse_0.0016/model.keras',
    help="Path to a saved .keras model checkpoint."
)
parser.add_argument(
    '--output-path',
    type=str,
    default='data/processed/spy_with_sentiment_with_predictions.csv',
    help="Where to save the dataset with predictions appended."
)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.data_path)

# Specify the columns to use
columns = [
    "CLOSE", "VOLUME", "OPEN", "HIGH", "LOW",
    "MACD", "MACD_Signal", "MACD_Hist", "RSI_10",
    "BB_Std", "sentiment_score", "cluster"
]

# Initialize sequence generator and generate sequences from full data
seq_len = 100
seq_gen = SequenceGenerator(df, seq_len=seq_len, cols=columns, normalise=True)

X, y_scaled = seq_gen.generate_sequences()

# Load your trained model
model = Model()
model.load_model(args.model_path)

# Make predictions
y_pred_scaled = model.predict_point_by_point(X)

# Unnormalize both predictions and actual values
y_pred_unnorm = seq_gen.inverse_target(y_pred_scaled)
y_test_unnorm = seq_gen.inverse_target(y_scaled)

# Plot predictions vs actual (on unnormalized/original scale)
model.plot_predictions(
    y_true=y_test_unnorm,
    y_pred=y_pred_unnorm,
    title="LSTM Predictions vs Actual (Full Test Data, Unnormalized)"
)

# Create a new column with NaNs for alignment
df['y_pred'] = [None] * len(df)

# Assign predictions starting from row `seq_len`
df.loc[seq_len:, 'y_pred'] = y_pred_unnorm

# Drop the first `seq_len` rows (no prediction possible for them)
df_final = df.iloc[seq_len:].reset_index(drop=True)

# Optional: Save or print
print(df_final)
df_final.to_csv(args.output_path, index=False)