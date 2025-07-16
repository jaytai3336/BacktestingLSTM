import os
import json
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, RMSprop, SGD
from core.utils import Timer
from datetime import datetime
from keras.losses import get as get_loss

# Disable oneDNN optimizations in TensorFlow for potential compatibility reasons
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class Model:
    def __init__(self):
        # Initialize an empty Sequential Keras model
        self.model = Sequential()
        self.configs = None
        self.history = None

    # ======= Model Construction and Compilation =======

    def build_optimizer(self, optimizer_name, learning_rate):
        """
        Returns optimizer instance based on name and learning rate.
        Supports Adam, RMSprop, SGD.
        """
        optimizers = {
            'adam': Adam,
            'rmsprop': RMSprop,
            'sgd': SGD
        }
        optimizer_cls = optimizers.get(optimizer_name.lower())
        if not optimizer_cls:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer_cls(learning_rate=learning_rate)

    def build_model(self, configs):
        """
        Build the model architecture based on a config dict.
        Config must contain 'model' and 'layers' keys.
        Supports layers: dense, lstm, dropout, bidirectional, conv1d.
        Adds Input layer explicitly on first layer.
        Compiles the model with specified loss, optimizer, and metrics.
        """
        if 'model' not in configs or 'layers' not in configs['model']:
            raise ValueError("Config must contain 'model' and 'layers' keys")

        timer = Timer()
        timer.start()
        self.configs = configs

        self.model = Sequential()
        first_layer = True

        for layer in configs['model']['layers']:
            layer_type = layer['type']
            neurons = layer.get('neurons')
            dropout_rate = layer.get('rate')
            activation = layer.get('activation')
            return_seq = layer.get('return_seq', False)
            input_timesteps = layer.get('input_timesteps')
            input_dim = layer.get('input_dim')

            # Add Input layer explicitly once at the start with shape (timesteps, features)
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

        optimizer_name = configs['model'].get('optimizer', 'adam')
        learning_rate = configs['model'].get('learning_rate', 0.001)
        optimizer = self.build_optimizer(optimizer_name, learning_rate)

        # Compile the model with loss, optimizer, and evaluation metrics
        self.model.compile(
            loss=configs['model']['loss'],
            optimizer=optimizer,
            metrics=configs['model']['metrics']
        )
        print('[Model] Model Compiled')
        timer.stop()

    # ======= Model Loading =======

    def load_model(self, filepath):
        """
        Load a pre-trained Keras model from file.
        Raises FileNotFoundError if file doesn't exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        print(f'[Model] Loading model from file {filepath}')
        self.model = load_model(filepath)

    # ======= Training Methods =======

    def train(self, x, y, epochs, batch_size, save_dir, validation_data=None):
        """
        Train the model on given data.
        Validates input shapes and values before training.
        Saves best model during training using EarlyStopping and ModelCheckpoint callbacks.
        """
        self._validate_training_data(x, y)

        # Confirm input data shape matches model input shape
        if x.shape[1:] != self.model.input_shape[1:]:
            raise ValueError(f"Input shape {x.shape[1:]} does not match model input shape {self.model.input_shape[1:]}")

        timer = Timer()
        timer.start()

        print(f'[Model] Training Started: {epochs} epochs, batch size {batch_size}')
        # Create unique filename for saving the model based on timestamp and epochs
        save_fname = os.path.join(
            save_dir, 'archive',
            f"{dt.datetime.now():%d%m%Y-%H%M%S}-e{self.configs['training']['epochs']}.keras"
        )

        callbacks = self._get_training_callbacks(save_fname)

        # Train the model
        self.history = self.model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data
        )

        print(f'[Model] Training Completed. Model saved as {save_fname}')
        # Load best weights from saved file
        self.model.load_weights(save_fname)
        print(self.model.summary())
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        """
        Train model using a data generator.
        Useful for large datasets or online data streams.
        Saves best model based on monitored metric.
        """
        timer = Timer()
        timer.start()
        print(f'[Model] Training Started: {epochs} epochs, batch size {batch_size}, steps per epoch {steps_per_epoch}')
        save_fname = os.path.join(
            save_dir, 'archive',
            f"{dt.datetime.now():%d%m%Y-%H%M%S}-e{epochs}-generator.keras"
        )

        callbacks = self._get_training_callbacks(save_fname)

        self.history = self.model.fit(
            data_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks
        )

        print(f'[Model] Training Completed. Model saved as {save_fname}')
        self.model.load_weights(save_fname)
        timer.stop()

    def _validate_training_data(self, x, y):
        """
        Utility to check if training data contains NaNs or Infs,
        which can cause training errors.
        """
        assert not np.isnan(x).any(), "Input x contains NaNs"
        assert not np.isnan(y).any(), "Target y contains NaNs"
        assert not np.isinf(x).any(), "Input x contains Infs"
        assert not np.isinf(y).any(), "Target y contains Infs"

    def _get_training_callbacks(self, save_fname):
        """
        Returns standard callbacks for training:
        EarlyStopping, ModelCheckpoint (saving best model), TensorBoard.
        Parameters such as patience and metric come from configs.
        """
        early_stopping_patience = self.configs['training'].get('early_stopping_patience', 3)
        monitor_metric = self.configs['training'].get('monitor_metric', 'loss')
        return [
            EarlyStopping(monitor=monitor_metric, patience=early_stopping_patience),
            ModelCheckpoint(filepath=save_fname, monitor=monitor_metric, save_best_only=True),
            TensorBoard(log_dir=os.path.join(os.path.dirname(save_fname), 'logs'))
        ]
    
    def train_with_teacher_forcing(self, x, y, epochs, batch_size, save_dir, validation_data=None, teacher_forcing_ratio=1.0, scheduled_sampling_decay=0.95):
        """
        Custom training loop with teacher forcing and optional scheduled sampling.
        teacher_forcing_ratio: probability of using ground truth as input.
        scheduled_sampling_decay: decay factor for reducing teacher forcing over epochs.
        """
        import tensorflow as tf

        timer = Timer()
        timer.start()

        print(f'[Model] Custom Training (Teacher Forcing): {epochs} epochs, batch size {batch_size}')
        optimizer = self.model.optimizer
        loss_fn = self.model.loss
        metric_fns = [tf.keras.metrics.get(m) for m in self.configs['model']['metrics']]

        # Prepare dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            epoch_loss = []
            [m.reset_state() for m in metric_fns]

            for step, (x_batch, y_batch) in enumerate(dataset):
                batch_size_tf = tf.shape(x_batch)[0]
                with tf.GradientTape() as tape:
                    decoder_input = x_batch[:, 0:1, :]  # first timestep
                    outputs = []

                    for t in range(1, y_batch.shape[1]):
                        output = self.model(decoder_input, training=True)
                        outputs.append(output)

                        use_teacher_forcing = np.random.rand() < teacher_forcing_ratio
                        next_input = y_batch[:, t:t+1, :] if use_teacher_forcing else output
                        decoder_input = next_input

                    outputs_tensor = tf.concat(outputs, axis=1)
                    loss_fn = get_loss(loss_fn)
                    loss_value = loss_fn(y_batch[:, 1:, :], outputs_tensor)

                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_loss.append(loss_value.numpy())

                for metric_fn in metric_fns:
                    metric_fn.update_state(y_batch[:, 1:, :], outputs_tensor)

            print(f"Loss: {np.mean(epoch_loss):.4f}", end=" ")
            for metric_fn in metric_fns:
                print(f"{metric_fn.name}: {metric_fn.result().numpy():.4f}", end=" ")
            print()

            # Decay the teacher forcing ratio
            teacher_forcing_ratio = max(0.05, teacher_forcing_ratio * scheduled_sampling_decay)

        print('[Model] Custom training completed.')
        timer.stop()

    # ======= Prediction Methods =======

    def predict(self, data):
        """
        Predict output for each input sequence individually (one-to-one).
        Returns flattened 1D array if single output.
        """
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        return np.reshape(predicted, (predicted.size,))  # flatten if single output

    def predict_bayesian_inference(self, data, n_iter=100):
        """
        Monte Carlo Dropout prediction: run the model multiple times
        with dropout enabled at inference to estimate mean and std dev.
        """
        print('[Model] Predicting with uncertainty...')
        predictions = [
            self.model(data, training=True).numpy() for _ in tqdm(range(n_iter), 'Predicting with uncertainty')
        ]
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def predict_rolling_forecasts(self, data, window_size, prediction_len):
        """
        Predict multiple sequences using rolling windows.
        For each prediction_len block, predict step-by-step feeding back predictions.
        Useful for multi-step forecasting.
        """
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        n_features = data.shape[2]  # assuming data shape is (num_samples, window_size, n_features)

        for i in tqdm(range(int(len(data) / prediction_len)), 'Predicting multiple sequences'):
            curr_frame = data[i * prediction_len].copy()  # shape (window_size, n_features)

            predicted = []
            for j in range(prediction_len):
                pred = self.model.predict(curr_frame[np.newaxis, :, :], verbose=0)[0, 0]
                predicted.append(pred)

                # Create full feature vector for predicted timestep
                pred_full = np.zeros((1, n_features))
                pred_full[0, 0] = pred  # assuming target is first feature

                # Shift window and append predicted timestep
                curr_frame = np.concatenate((curr_frame[1:], pred_full), axis=0)

            prediction_seqs.append(predicted)

        return prediction_seqs

    def predict_recursive_forecasts(self, data, window_size):
        """
        Predict a full sequence by iteratively feeding the predicted output
        back as input for the next timestep.
        """
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]  # shape (window_size, n_features)
        predicted = []
        for _ in tqdm(range(len(data)), 'Predicting full sequence'):
            pred = self.model.predict(curr_frame[np.newaxis, :, :], verbose=0)[0, 0]
            predicted.append(pred)
            pred_full = np.zeros((1, curr_frame.shape[1]))
            pred_full[0, 0] = pred
            curr_frame = np.concatenate((curr_frame[1:], pred_full), axis=0)
        return predicted

    # ======= Evaluation =======

    def evaluate(self, x, y, metrics=['mse', 'mae'], save=False):
        """
        Evaluate model performance on test data.
        Prints requested metrics.
        Optionally saves the model and training artifacts.
        """
        print('[Model] Evaluating model...')
        results = self.model.evaluate(x, y, return_dict=True)
        for metric in metrics:
            print(f'[Model] {metric}: {results.get(metric, "N/A")}')

        if save:
            folder_name = '_'.join(
                [datetime.now().strftime("%m%d_%H%M%S")] +
                [f"{metric}_{results.get(metric, 0):.4f}" for metric in metrics]
            )
            self.save_all(folder_name)

        return results

    # ======= Plotting =======

    def plot_predictions(self, y_true, y_pred, title="Predictions vs Actual"):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="Actual", linewidth=2)
        plt.plot(y_pred, label="Predicted", linestyle='--')
        plt.title(title)
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    # ======= Weight Management =======

    def get_weights_by_layer(self, layer_name=None):
        """
        Retrieve weights and biases for all layers or a specific layer.
        Returns dictionary of layer_name: [weights, biases].
        """
        print('[Model] Retrieving weights...')
        weights_dict = {}
        for layer in self.model.layers:
            if layer_name is None or layer.name == layer_name:
                weights = layer.get_weights()
                if weights:
                    weights_dict[layer.name] = weights
                if layer_name and layer.name == layer_name:
                    return weights
        if layer_name and layer_name not in weights_dict:
            raise ValueError(f"Layer {layer_name} not found in model")
        return weights_dict if layer_name is None else []

    def set_weights_by_layer(self, layer_name, weights):
        """
        Set weights and biases for a specific layer.
        Checks that provided weights shape matches layer weights.
        """
        print(f'[Model] Setting weights for layer {layer_name}...')
        for layer in self.model.layers:
            if layer.name == layer_name:
                expected_shapes = [w.shape for w in layer.get_weights()]
                provided_shapes = [w.shape for w in weights]
                if expected_shapes != provided_shapes:
                    raise ValueError(f"Weight shape mismatch for layer {layer_name}. Expected {expected_shapes}, got {provided_shapes}")
                layer.set_weights(weights)
                return
        raise ValueError(f"Layer {layer_name} not found in model")

    def save_weights(self, dir):
        """
        Save model weights to a file named '.weights.h5' in the given directory.
        """
        save_fname = os.path.join(dir, '.weights.h5')
        print(f'[Model] Saving weights to {save_fname}...')
        self.model.save_weights(save_fname)
        print('[Model] Weights saved successfully.')

    def load_weights(self, filepath):
        """
        Load model weights from a given file.
        Raises error if file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        print(f'[Model] Loading weights from {filepath}...')
        self.model.load_weights(filepath)
        print('[Model] Weights loaded successfully.')

    # ======= Saving Model and History =======

    def save_model_summary(self, dir):
        """
        Save the textual model summary to 'model_summary.txt' in given directory.
        """
        save_fname = os.path.join(dir, 'model_summary.txt')
        print(f'[Model] Saving model summary to {dir}...')
        with open(save_fname, 'w', encoding='utf-8') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print('[Model] Model summary saved successfully.')

    def save_model_json(self, dir):
        """
        Save the model architecture JSON to 'model_architecture.json' in given directory.
        """
        save_fname = os.path.join(dir, 'model_architecture.json')
        print(f'[Model] Saving model json to {dir}...')
        with open(save_fname, "w") as f:
            f.write(self.model.to_json())
        print('[Model] Model json saved successfully.')

    def save_history(self, dir):
        """
        Save the training history (loss, metrics per epoch) as JSON to 'training_history.json'.
        """
        save_fname = os.path.join(dir, 'training_history.json')
        print(f'[Model] Saving training history to {dir}...')
        with open(save_fname, 'w') as f:
            json.dump(self.history.history, f)
        print('[Model] Training history saved successfully.')

    def save_all(self, foldername):
        """
        Save all model artifacts:
        summary, architecture json, weights, full model, and training history.
        """
        if not self.configs or 'model' not in self.configs or 'save_dir' not in self.configs['model']:
            raise ValueError("Model save directory not found in configs")

        dir = os.path.join(self.configs['model']['save_dir'], foldername)
        os.makedirs(dir, exist_ok=True)

        self.save_model_summary(dir)
        self.save_model_json(dir)
        self.save_weights(dir)
        self.model.save(os.path.join(dir, 'model.keras'))
        self.save_history(dir)

from keras.callbacks import Callback

class CustomEarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=5, min_delta=0, cooldown=0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.wait = 0
        self.best = None
        self.cooldown_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            self.cooldown_counter = 0
        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Stopping early at epoch {epoch+1}")
                    self.model.stop_training = True
