import numpy as np
from core.model import Model

# Main script
def main():
    # Example configuration
    configs = {
        'model': {
            'loss': 'mae',
            'optimizer': 'adam',
            'layers': [
                {'type': 'lstm', 'neurons': 50, 'input_timesteps': 10, 'input_dim': 1, 'return_seq': False},
                {'type': 'dense', 'neurons': 1, 'activation': 'linear'}
            ]
        }
    }

    # Initialize and build model
    model = Model()
    model.build_model(configs)

    # Get weights for all layers
    all_weights = model.get_weights_by_layer()
    for layer_name, weights in all_weights.items():
        print(f"Layer {layer_name}: Weight shape {weights[0].shape}, Bias shape {weights[1].shape}")

    # Get weights for a specific layer
    lstm_weights = model.get_weights_by_layer(layer_name='lstm')
    print(f"LSTM weights shape: {lstm_weights[0].shape}, Biases shape: {lstm_weights[1].shape}")

    # Modify weights (example: set weights to zeros for a specific layer)
    new_weights = [np.zeros_like(w) for w in lstm_weights]
    model.set_weights_by_layer(layer_name='lstm', weights=new_weights)

if __name__ == '__main__':
    main()