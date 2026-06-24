import numpy as np
import math
import pandas as pd
import os
from core.model import Model
from core.utils import Timer
from core.preprocessing import DataLoader
import json

def main():
    # Load configuration
    configs = json.load(open(os.path.join('Lstm', 'config.json'), 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    timer = Timer()
    timer.start()

    # Initialize DataLoader
    data_loader = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # Create sequences for training and testing
    x_train, y_train = data_loader.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    x_test, y_test = data_loader.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    timer.stop()

    # Step 2: Build model
    print("\nBuilding model...")
    model = Model()
    model.build_model(configs)

    # Load weights
    # model.load_weights()

    # Step 3: Train using using train
    print("\nTraining model...")
    if configs['training']['generator'] == False:
        model.train(x_train, y_train, epochs=configs['training']['epochs'], batch_size=configs['training']['batch_size'], save_dir=configs['model']['save_dir'])
    else:
        steps_per_epoch = math.ceil((data_loader.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
        train_generator = data_loader.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        )
        model.train_generator(
            data_gen=train_generator,
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir']
        )

    # Check weights after training
    print("\nWeights after training (LSTM layer, first 5 values):")
    weights_after = model.get_weights_by_layer('lstm')
    print(weights_after[0].flatten()[:5])

    # Step 4: Modify weights (example: scale LSTM weights by 1.1)
    print("\nModifying LSTM weights (scaling by 1.1)...")
    modified_weights = [w * 1.1 for w in weights_after]
    model.set_weights_by_layer('lstm', modified_weights)

    # Verify modified weights
    print("\nWeights after modification (LSTM layer, first 5 values):")
    weights_modified = model.get_weights_by_layer('lstm')
    print(weights_modified[0].flatten()[:5])

    # Step 5: Evaluate model
    print("\nEvaluating model...")
    # y_pred = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # y_pred = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    y_pred = model.predict_point_by_point(x_test)

    # y_pred and y_test are currently normalized
    y_pred_unnorm = data_loader.scaler_target.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    y_test_unnorm = data_loader.scaler_target.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()


    # Evaluate on original scale
    model.evaluate(x_test, y_test, metrics=configs['model']['metrics'])

    # Plot on original scale
    model.plot_predictions(
        y_true=y_test_unnorm,
        y_pred=y_pred_unnorm,
        title="LSTM Predictions vs Actual (Test Data, Unnormalized)"
    )

    # Save original scale predictions
    results = pd.DataFrame({
        'Actual': y_test_unnorm,
        'Predicted': y_pred_unnorm
    })
    results.to_csv(os.path.join(configs['model']['save_dir'], 'predictions.csv'), index=False)
    print(f"Predictions saved to {os.path.join(configs['model']['save_dir'], 'predictions.csv')}")

if __name__ == '__main__':
    main()