#!/usr/bin/env python3

from extract import extract
from metrics import f1_score

import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import sys
import pickle


def build_model(input_dim, output_dim, embedding_layer):
    sequence_input = Input(shape=(input_dim, ))
    embedded_sequence = embedding_layer(sequence_input)
    x = GRU(512, dropout=0.5, return_sequences=False)(embedded_sequence)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    prediction = Dense(output_dim, activation="sigmoid")(x)

    model = Model(sequence_input, prediction)

    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=[f1_score])

    return model


if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # set_session(tf.Session(config=config))

    training_file_name = sys.argv[1]
    testing_file_name = sys.argv[2]
    wordvec_file_name = sys.argv[3]

    training_x, training_y, word_index, testing_x, classes = extract(
        training_file_name, testing_file_name)
    training_x = np.array(training_x)
    training_y = np.array(training_y)

    wordvec = {}
    with open(wordvec_file_name, "r") as wordvec_file:
        for line in wordvec_file:
            columns = line.split(" ")
            wordvec[columns[0]] = np.array(
                [float(n) for n in columns[1:]], dtype="float32")

    indices = np.arange(training_x.shape[0])
    np.random.seed(19940622)
    np.random.shuffle(indices)
    training_x = training_x[indices]
    training_y = training_y[indices]
    num_validating_x = training_x.shape[0] // 5

    validating_x = training_x[-num_validating_x:]
    validating_y = training_y[-num_validating_x:]
    training_x = training_x[:-num_validating_x]
    training_y = training_y[:-num_validating_x]

    num_words = len(word_index) + 1
    wordvec_dimension = 300  # glove
    embedding_matrix = np.zeros((num_words, wordvec_dimension))
    for word, i in word_index.items():
        if word in wordvec.keys():
            embedding_matrix[i] = wordvec[word]
        if word.capitalize() in wordvec.keys():
            embedding_matrix[i] = wordvec[word.capitalize()]

    sequence_length = training_x.shape[1]
    num_classes = training_y.shape[1]
    embedding_layer = Embedding(
        num_words,
        wordvec_dimension,
        input_length=sequence_length,
        weights=[embedding_matrix],
        trainable=False)

    model = build_model(sequence_length, num_classes, embedding_layer)
    model.summary()

    # class_weight = {
    #     k: v
    #     for k, v in enumerate(1 / np.mean(training_y, axis=0))
    # }

    history = model.fit(
        training_x,
        training_y,
        validation_data=(validating_x, validating_y),
        batch_size=128,
        epochs=100,
        # class_weight=class_weight,
        callbacks=[
            EarlyStopping(
                monitor="val_f1_score", mode="max", patience=10, verbose=1),
            ModelCheckpoint(
                "model.h5",
                save_best_only=True,
                save_weights_only=True,
                monitor="val_f1_score",
                mode="max",
                verbose=1),
            # ReduceLROnPlateau(
            #         monitor="val_f1_score", factor=0.5, patience=5, verbose=1)
        ])

    with open("history.p", "wb") as history_file:
        pickle.dump(history.history, history_file)
