# Copyright (c) 2023 Zander Raycraft
# DISCLAIMER: The training data was provided by Kaggle, linked below
# https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data#

import pandas as pd
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def parseCSV(filePath):
    data = pd.read_csv(filePath)
    data['pixels'] = data['pixels'].apply(lambda x: list(map(int, x.split())))
    X = np.array(data['pixels'].tolist()).reshape(-1, 48, 48, 1)
    X = X / 255.0
    y = pd.get_dummies(data['emotion']).values
    return X, y

def construct_network():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def save_keras_model(model):
    base_filename = '../model/emotionIndicatorV'
    version = 1
    while os.path.exists(f"{base_filename}{version}.keras"):
        version += 1
    savedName = f"{base_filename}{version}.keras"
    model.save(savedName)
    return version, savedName

def convert_to_frozen_graph(model, path, version):
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    frozen_graph_dir = '../frozen_graph'
    frozen_graph_filename = f"emotionIndicatorFrozenGraphV{version}"
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=frozen_graph_dir, name=frozen_graph_filename, as_text=False)
    return os.path.join(frozen_graph_dir, frozen_graph_filename)

def main():
    time_started = time.time()

    print("-- Beginning CSV parsing of Data -- ")
    xData, yData = parseCSV('../data/train.csv')
    parseTime = "{:.5f}".format(time.time() - time_started)
    print(f"-- Successfully parsed data in: {parseTime} sec -- \n")

    print("-- Beginning construction of the Neural Network -- ")
    constructStartTime = time.time()
    model = construct_network()
    constructTime = "{:.5f}".format(time.time() - constructStartTime)
    print(f"-- Successfully constructed Neural Network: {constructTime} seconds --\n")

    modelFitStartTime = time.time()
    print("-- Beginning fitting of the Neural Network on training data-- ")
    model.fit(xData, yData, epochs=20, batch_size=32)
    finalFitTime = "{:.5f}".format(time.time() - modelFitStartTime)
    print(f"-- Successfully fitted data to model: {finalFitTime} seconds --\n")

    endTime = time.time()
    execTime = time_started - endTime
    print("Model summary: \n")
    model.summary()
    print("__________________\n")
    print(f"-- Total execution time: {execTime} sec --")

    version, savedName = save_keras_model(model)
    print(f"-- Successfully trained model V{version}")
    print(f"-- Saved model path is: {savedName}")

    frozen_graph_path = convert_to_frozen_graph(model, '', version)
    print(f"-- Converted model to frozen graph and saved at: {frozen_graph_path} --")

if __name__ == "__main__":
    main()
