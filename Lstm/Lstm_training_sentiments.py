import os
import json
import math
import numpy as np
import pandas as pd
from core.model import Model
from core.utils import Timer
from core.preprocessing import TimeSeriesDataLoader


def load_config(path='Lstm/config2.json'):
    with open(path, 'r') as f:
        return json.load(f)


def prepare_data(configs):
    print("Loading and preprocessing data...")
    timer = Timer()
    timer.start()

    dataset_path = os.path.join('data', configs['data']['filename'])
    data_loader = TimeSeriesDataLoader(
        filename=dataset_path,
        train_test_split=configs['data']['train_test_split'],
        cols=configs['data']['columns']
    )

    seq_len = configs['data']['sequence_length']
    normalise = configs['data']['normalise']

    x_train, y_train = data_loader.get_train_data(seq_len=seq_len, normalise=normalise)
    x_test, y_test = data_loader.get_test_data(seq_len=seq_len, normalise=normalise)

    timer.stop()
    return data_loader, x_train, y_train, x_test, y_test


def initialize_model(configs):
    model = Model()
    model.build_model(configs)

    model_path = os.path.join(configs['model']['save_dir'], 'model.keras')
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_model(model_path)
    else:
        print(f"No saved model found at {model_path}. A new model will be initialized.")

    return model


def train_model(model, data_loader, x_train, y_train, x_test, y_test, configs):
    print("\nTraining model...")
    train_cfg = configs['training']
    seq_len = configs['data']['sequence_length']
    batch_size = train_cfg['batch_size']

    if train_cfg.get('teaching', False):
        model.train_with_teacher_forcing(
            x_train, y_train,
            epochs=train_cfg['epochs'],
            batch_size=batch_size,
            save_dir=configs['model']['save_dir'],
            validation_data=(x_test, y_test),
            teacher_forcing_ratio=1.0,
            scheduled_sampling_decay=0.95
        )

    elif train_cfg.get('generator', False):
        steps_per_epoch = math.ceil((data_loader.len_train - seq_len) / batch_size)
        train_gen = data_loader.generate_train_batch(seq_len=seq_len, batch_size=batch_size, normalise=configs['data']['normalise'])
        model.train_generator(
            data_gen=train_gen,
            epochs=train_cfg['epochs'],
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir']
        )

    else:
        model.train(
            x_train, y_train,
            epochs=train_cfg['epochs'],
            batch_size=batch_size,
            save_dir=configs['model']['save_dir'],
            validation_data=(x_test, y_test)
        )


def make_predictions(model, x_test, y_test, data_loader, configs):
    print("\nGenerating predictions...")
    method = configs['model'].get('prediction_method', 'default')
    seq_len = configs['data']['sequence_length']

    if method == 'multiple':
        y_pred = model.predict_sequences_multiple(x_test, seq_len, seq_len)
    elif method == 'bayesian':
        y_pred = model.predict_bayesian_inference(x_test)
    elif method == 'recursive':
        y_pred = model.predict_recursive_forecasts(x_test, seq_len)
    else:
        y_pred = model.predict(x_test)

    # Unnormalize predictions
    y_pred_unnorm = data_loader.scaler_target.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    y_test_unnorm = data_loader.scaler_target.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()

    # Evaluate and plot
    model.evaluate(x_test, y_test, metrics=configs['model']['metrics'], save=True)
    model.plot_predictions(y_test_unnorm, y_pred_unnorm, title="LSTM Predictions vs Actual (Test Data)")

    # Save predictions
    results = pd.DataFrame({'Actual': y_test_unnorm, 'Predicted': y_pred_unnorm})
    output_path = os.path.join(configs['model']['save_dir'], 'test_predictions.csv')
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    configs = load_config()
    os.makedirs(configs['model']['save_dir'], exist_ok=True)

    mode = configs.get("mode", "train").lower()
    print(f"Running in {mode.upper()} mode...")

    data_loader, x_train, y_train, x_test, y_test = prepare_data(configs)
    model = initialize_model(configs)

    if mode == 'train':
        train_model(model, data_loader, x_train, y_train, x_test, y_test, configs)

    make_predictions(model, x_test, y_test, data_loader, configs)


if __name__ == '__main__':
    main()
