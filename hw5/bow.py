#!/usr/bin/env python3

from configs import num_models, wordvec_dimension
from extract import extract
from metrics import f1_score

import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
import pickle


def build_model(input_dim, output_dim):
    bow_input = Input(shape=(input_dim, ))
    x = Dense(1024, activation="relu")(bow_input)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    prediction = Dense(output_dim, activation="sigmoid")(x)

    model = Model(bow_input, prediction)

    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=[f1_score])

    return model


if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # set_session(tf.Session(config=config))

    np.random.seed(19940622)

    training_file_name = sys.argv[1]
    testing_file_name = sys.argv[2]

    training_x, training_y, tokenizer, testing_x, classes = extract(
        training_file_name, testing_file_name)
    training_x = np.array(tokenizer.sequences_to_matrix(training_x.tolist()))
    training_y = np.array(training_y)

    validating_ratio = 0.3
    num_validating_x = int(training_x.shape[0] * validating_ratio)
    num_words = training_x.shape[1]
    num_classes = training_y.shape[1]
    indices = np.arange(training_x.shape[0])
    model = build_model(num_words, num_classes)
    model.summary()

    np.random.shuffle(indices)
    training_x = training_x[indices]
    training_y = training_y[indices]
    new_training_x = training_x[:-num_validating_x]
    new_training_y = training_y[:-num_validating_x]
    new_validating_x = training_x[-num_validating_x:]
    new_validating_y = training_y[-num_validating_x:]

    history = model.fit(
        new_training_x,
        new_training_y,
        validation_data=(new_validating_x, new_validating_y),
        batch_size=512,
        epochs=100,
        callbacks=[
            EarlyStopping(
                monitor="val_f1_score", mode="max", patience=5, verbose=1),
            ModelCheckpoint(
                "model_bow.h5",
                save_best_only=True,
                save_weights_only=True,
                monitor="val_f1_score",
                mode="max",
                verbose=1),
        ])

    with open("history_bow.p", "wb") as history_file:
        pickle.dump(history.history, history_file)
