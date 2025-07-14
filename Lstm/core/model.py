import numpy as np
import datetime as dt
import json
from keras import Input
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.utils import Timer

import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class Model:
    def __init__(self):
        self.model = Sequential()
        self.configs = None
        self.history = None

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_optimizer(self, optimizer_name, learning_rate):
        if optimizer_name.lower() == 'adam':
            return Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            return RMSprop(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            return SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def build_model(self, configs):
        if 'model' not in configs or 'layers' not in configs['model']:
            raise ValueError("Config must contain 'model' and 'layers' keys")
        
        timer = Timer()
        timer.start()
        self.configs = configs

        first_layer = True

        for layer in configs['model']['layers']:
            layer_type = layer['type']
            neurons = layer.get('neurons')
            dropout_rate = layer.get('rate')
            activation = layer.get('activation')
            return_seq = layer.get('return_seq', False)
            input_timesteps = layer.get('input_timesteps')
            input_dim = layer.get('input_dim')

            # Add input layer explicitly once at the beginning
            if first_layer and input_timesteps and input_dim:
                self.model.add(Input(shape=(input_timesteps, input_dim)))
                first_layer = False

            if layer_type == 'dense':
                self.model.add(Dense(neurons, activation=activation))

            elif layer_type == 'lstm':
                self.model.add(LSTM(neurons, return_sequences=return_seq))

            elif layer_type == 'dropout':
                self.model.add(Dropout(dropout_rate))

            elif layer_type == 'bidirectional':
                self.model.add(Bidirectional(LSTM(neurons, return_sequences=return_seq)))

            elif layer_type == 'conv1d':
                filters = layer.get('filters', 64)
                kernel_size = layer.get('kernel_size', 3)
                self.model.add(Conv1D(filters, kernel_size, activation=activation))

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            
        optimizer_config = configs['model'].get('optimizer', 'adam')
        learning_rate = configs['model'].get('learning_rate', 0.001)
        optimizer = self.build_optimizer(optimizer_config, learning_rate)

        self.model.compile(loss=configs['model']['loss'], optimizer=optimizer, metrics=configs['model']['metrics'])
        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir, validation_data=None):
        assert not np.isnan(x).any(), "Input x contains NaNs"
        assert not np.isnan(y).any(), "Target y contains NaNs"
        assert not np.isinf(x).any(), "Input x contains Infs"
        assert not np.isinf(y).any(), "Target y contains Infs"

        if x.shape[1:] != self.model.input_shape[1:]:
            raise ValueError(f"Input shape {x.shape[1:]} does not match model input shape {self.model.input_shape[1:]}")
        
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        save_fname = os.path.join(save_dir, 'archive/%s-e%s.keras' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(self.configs['training']['epochs'])))
        
        early_stopping_patience = self.configs['training'].get('early_stopping_patience', 3)
        monitor_metric = self.configs['training'].get('monitor_metric', 'loss')
        callbacks = [
            EarlyStopping(monitor=monitor_metric, patience=early_stopping_patience),
            ModelCheckpoint(filepath=save_fname, monitor=monitor_metric, save_best_only=True),
            TensorBoard(log_dir=os.path.join(save_dir, 'logs'))
        ]

        self.history = self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        self.model.load_weights(save_fname)
        print(self.model.summary())
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        save_fname = os.path.join(save_dir, 'archive/%s-e%s-generator.keras' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        
        early_stopping_patience = self.configs['training'].get('early_stopping_patience', 3)
        monitor_metric = self.configs['training'].get('monitor_metric', 'loss')
        callbacks = [
            EarlyStopping(monitor=monitor_metric, patience=early_stopping_patience),
            ModelCheckpoint(filepath=save_fname, monitor=monitor_metric, save_best_only=True),
            TensorBoard(log_dir=os.path.join(save_dir, 'logs'))
        ]

        self.history = self.model.fit(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks
        )
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        self.model.load_weights(save_fname)
        timer.stop()

    def predict_point_by_point(self, data):
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))  # Comment if multi output
        return predicted
    
    def predict_with_uncertainty(self, data, n_iter=100):
        print('[Model] Predicting with uncertainty...')
        predictions = []
        for _ in tqdm(range(n_iter), 'Predicting with uncertainty'):
            # Use model.__call__ with training=True to enable dropout during inference
            pred = self.model(data, training=True)
            predictions.append(pred.numpy())  # Convert tensor to numpy array
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in tqdm(range(int(len(data)/prediction_len)), 'Predicting multiple sequences'):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[np.newaxis,:,:], verbose=0)[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in tqdm(range(len(data)), 'Predicting full sequence'):
            predicted.append(self.model.predict(curr_frame[np.newaxis,:,:], verbose=0)[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted

    def evaluate(self, x, y, metrics=['mse', 'mae']):
        print('[Model] Evaluating model...')
        results = self.model.evaluate(x, y, return_dict=True)
        for metric in metrics:
            print(f'[Model] {metric}: {results.get(metric, "N/A")}')

        # save results
        folder_name_parts = [f"{metric}_{results.get(metric, 0):.4f}" for metric in metrics]
        folder_name = '_'.join(folder_name_parts)
        self.save_all(folder_name)
    
        return results
    
    def plot_predictions(self, y_true, y_pred, title="Predictions vs Actual"):
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.title(title)
        plt.legend()
        plt.show()

    def get_weights_by_layer(self, layer_name=None):
        """
        Retrieve weights and biases for all layers or a specific layer.
        
        Args:
            layer_name (str, optional): Name of the specific layer to get weights for.
                                       If None, returns weights for all layers.
        
        Returns:
            dict or list: If layer_name is specified, returns a list of numpy arrays [weights, biases].
                          If layer_name is None, returns a dict with layer names as keys and
                          lists of [weights, biases] as values.
        """
        print('[Model] Retrieving weights...')
        weights_dict = {}
        for layer in self.model.layers:
            if layer_name is None or layer.name == layer_name:
                layer_weights = layer.get_weights()
                if layer_weights:  # Only include layers with weights
                    weights_dict[layer.name] = layer_weights
                if layer_name and layer.name == layer_name:
                    return layer_weights
        if layer_name and layer_name not in weights_dict:
            raise ValueError(f"Layer {layer_name} not found in model")
        return weights_dict if not layer_name else []

    def set_weights_by_layer(self, layer_name, weights):
        """
        Set weights and biases for a specific layer.
        
        Args:
            layer_name (str): Name of the layer to set weights for.
            weights (list): List of numpy arrays [weights, biases] matching the layer's weight shapes.
        
        Raises:
            ValueError: If layer_name is not found or weights shape doesn't match.
        """
        print(f'[Model] Setting weights for layer {layer_name}...')
        for layer in self.model.layers:
            if layer.name == layer_name:
                expected_shapes = [w.shape for w in layer.get_weights()]
                provided_shapes = [w.shape for w in weights]
                if expected_shapes != provided_shapes:
                    raise ValueError(
                        f"Weight shape mismatch for layer {layer_name}. "
                        f"Expected {expected_shapes}, got {provided_shapes}"
                    )
                layer.set_weights(weights)
                return
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def save_weights(self, dir):
        """
        Save the model's weights to a specified file.

        Args:
            filepath (str): Path where to save the weights file.
        """
        save_fname = os.path.join(dir, '.weights.h5')
        print(f'[Model] Saving weights to {save_fname}...')
        self.model.save_weights(save_fname)
        print('[Model] Weights saved successfully.')

    def load_weights(self, filepath):
        """
        Load model weights from a specified file.

        Args:
            filepath (str): Path of the weights file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        print(f'[Model] Loading weights from {filepath}...')
        self.model.load_weights(filepath)
        print('[Model] Weights loaded successfully.')       

    def save_model_summary(self, dir):
        """
        Save the model's architecture summary to a text file.
        """
        save_fname = os.path.join(dir, 'model_summary.txt')
        print(f'[Model] Saving model summary to {dir}...')
        with open(save_fname, 'w', encoding='utf-8') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print('[Model] Model summary saved successfully.')
    
    def save_model_json(self, dir):
        """
        Save the model's json summary to a text file.
        """
        save_fname = os.path.join(dir, 'model_architecture.json')
        print(f'[Model] Saving model json to {dir}...')
        with open(save_fname, "w") as f:
            f.write(self.model.to_json())
        print('[Model] Model json saved successfully.')

    def save_history(self, dir):
        """
        Save the model's history to a text file.
        """
        save_fname = os.path.join(dir, 'training_history.json')
        print(f'[Model] Saving training history to {dir}...')
        with open(os.path.join(dir, 'training_history.json'), 'w') as f:
            json.dump(self.history.history, f)       
        print('[Model] training history saved successfully.')       

    def save_all(self, foldername):
        if not self.configs or 'model' not in self.configs or 'save_dir' not in self.configs['model']:
            raise ValueError("Model save directory not found in configs")

        dir = os.path.join(self.configs['model']['save_dir'], foldername)
        os.makedirs(dir, exist_ok=True)

        self.save_model_summary(dir)
        self.save_model_json(dir)
        self.save_weights(dir)
        self.model.save(os.path.join(dir, 'model.keras'))
        self.save_history(dir)