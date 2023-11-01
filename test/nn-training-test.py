import pytest
import pandas as pd
import numpy as np
from NN_training import train
from src import cameraFeed
import io
import os
import tensorflow as tf
import pytest
from unittest.mock import Mock, patch
import cv2
import numpy as np


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


# creaes a mock model of the neural network
#
# @param none
# @return model, the mocked model to be used
@pytest.fixture
def mock_model():
    model = Mock()
    model.predict.return_value = np.array([[1, 0, 0, 0, 0, 0, 0]])  # Mock prediction for 'Angry'
    return model


# Mocks stream capture from camera
@pytest.fixture
def mock_capture():
    class MockCapture:
        def __init__(self, is_opened=True):
            self._is_opened = is_opened
        def isOpened(self):
            return self._is_opened
        def read(self):
            # Return a mock frame which is a black image and a ret value
            return True, np.zeros((48, 48, 3), dtype=np.uint8)
        def release(self):
            pass
    return MockCapture()
    
# Utilizes the made class to simulate a stream capture
#
# @param none
# @return none, but will raise errors and fail test is assertions fail
@pytest.fixture
def mock_video_capture(mock_capture):
    with patch('cv2.VideoCapture', return_value=mock_capture) as mocked_capture:
        yield mocked_capture


# Runs a test to see the stream frames can be preprocessed for the neural network
#
# @param none
# @return none, but will raise errors and fail test is assertions fail  
def test_preprocess_frame():
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    preprocessed_frame = cameraFeed.preprocess_frame(frame)
    assert preprocessed_frame.shape == (1, 48, 48, 1)
 

# Runs a test to see if the model can be properly loaded
#
# @param none
# @return none, but will raise errors and fail test is assertions fail  
def test_load_emotion_model(mock_model):
    model = cameraFeed.load_emotion_model('../model/emotionIndicatorV3.keras')
    assert model is not None   
    

# Runs a test to see the correct emotion can be displayed
#
# @param none
# @return none, but will raise errors and fail test is assertions fail  
def test_display_emotion():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    emotion_label = 'Happy'
    cameraFeed.display_emotion(frame, emotion_label)

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
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert model.loss == 'categorical_crossentropy'
    assert 'accuracy' in model.compiled_metrics._metrics

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
