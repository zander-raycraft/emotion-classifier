import pytest
import pandas as pd
import numpy as np
from NN_training import train
import io
import os
import tensorflow as tf

# Mock data and utilities
zeroes = " ".join(["0"] * 2304)
two_fifty_fives = " ".join(["255"] * 2304)
MOCK_CSV_DATA = f"""emotion,pixels
0,{zeroes}
1,{two_fifty_fives}
2,{zeroes}
3,{two_fifty_fives}
4,{zeroes}
5,{two_fifty_fives}
6,{zeroes}
"""


# establishes a mock_read to prevent an infinite recursion edge case for the CSV parser test - Helper function
#
# @param *args
# @params **kwargs
# @return none, but will raise errors and fail test is assertions fail
def mock_read_csv(*args, **kwargs):
    # Create DataFrame directly without using pd.read_csv
    data = {
        'emotion': [0, 1, 2, 3, 4, 5, 6],
        'pixels': [zeroes, two_fifty_fives, zeroes, two_fifty_fives, zeroes, two_fifty_fives, zeroes]
    }
    return pd.DataFrame(data)


# Tests the functionality of the CSV parser
#
# @param monkeypath - the mock csv data trained on
# @return none, but will raise errors and fail test is assertions fail
def test_parseCSV(monkeypatch):
    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)
    
    X, y = train.parseCSV('dummy_path')
    assert X.shape == (7, 48, 48, 1)
    assert y.shape == (7, 7)
    assert np.max(X) <= 1.0  # Checking normalization


# Tests to see network construction
#
# @param none
# @return none, but will raise errors and fail test is assertions fail
def test_construct_network():
    model = train.construct_network()
    assert model.layers[0].input_shape == (None, 48, 48, 1)
    assert len(model.layers) == 8
    # Check against Adam optimizer
    assert isinstance(model.optimizer, tf.keras.optimizers.legacy.Adam)
    assert model.loss == 'categorical_crossentropy'
    assert 'accuracy' in model.compiled_metrics._metrics


# Tests the functionality transforming a .keras file into a .pb frozen graph file
#
# @param none
# @return none, but will raise errors and fail test is assertions fail
def test_convert_to_frozen_graph():
    model = train.construct_network()
    version, _ = train.save_keras_model(model)
    frozen_graph_path = train.convert_to_frozen_graph(model, '', version)
    assert os.path.exists(frozen_graph_path)
    os.remove(frozen_graph_path)  # Clean up after test

# Tests the deeper detail of the construct of the model, such as layers, epochs, and flatte/convolution layers
#
# @param none
# @return none, but will raise errors and fail test is assertions fail
def test_detailed_network_structure():
    model = train.construct_network()

    # Check first convolution layer details
    assert isinstance(model.layers[0], tf.keras.layers.Conv2D)
    assert model.layers[0].filters == 64
    assert model.layers[0].kernel_size == (3, 3)
    assert model.layers[0].activation.__name__ == 'relu'

    # Check second pooling layer details
    assert isinstance(model.layers[1], tf.keras.layers.MaxPooling2D)
    assert model.layers[1].pool_size == (2, 2)

    # Check second convolution layer details
    assert isinstance(model.layers[2], tf.keras.layers.Conv2D)
    assert model.layers[2].filters == 128
    assert model.layers[2].kernel_size == (3, 3)
    assert model.layers[2].activation.__name__ == 'relu'

    # Check third pooling layer details
    assert isinstance(model.layers[3], tf.keras.layers.MaxPooling2D)
    assert model.layers[3].pool_size == (2, 2)

    # Check flatten layer
    assert isinstance(model.layers[4], tf.keras.layers.Flatten)

    # Check first dense layer details
    assert isinstance(model.layers[5], tf.keras.layers.Dense)
    assert model.layers[5].units == 512
    assert model.layers[5].activation.__name__ == 'relu'

    # Check dropout layer details
    assert isinstance(model.layers[6], tf.keras.layers.Dropout)
    assert model.layers[6].rate == 0.5

    # Check final dense layer details
    assert isinstance(model.layers[7], tf.keras.layers.Dense)
    assert model.layers[7].units == 7
    assert model.layers[7].activation.__name__ == 'softmax'
