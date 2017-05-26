#!/usr/bin/env python3

from configs import num_models, wordvec_dimension
from extract import extract
from metrics import f1_score

import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout
from keras.layers import GRU, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

    np.random.seed(19940622)

    training_file_name = sys.argv[1]
    testing_file_name = sys.argv[2]
    wordvec_file_name = sys.argv[3]

    training_x, training_y, tokenizer, testing_x, classes = extract(
        training_file_name, testing_file_name)
    training_x = np.array(training_x)
    training_y = np.array(training_y)
    word_index = tokenizer.word_index

    wordvec = {}
    with open(wordvec_file_name, "r") as wordvec_file:
        for line in wordvec_file:
            columns = line.split(" ")
            wordvec[columns[0]] = np.array(
                [float(n) for n in columns[1:]], dtype="float32")

    num_words = len(word_index) + 1
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

    validating_ratio = 0.3
    num_validating_x = int(training_x.shape[0] * validating_ratio)
    indices = np.arange(training_x.shape[0])
    historys = []
    for i in range(num_models):
        model = build_model(sequence_length, num_classes, embedding_layer)
        model.summary()

        np.random.shuffle(indices)
        training_x = training_x[indices]
        training_y = training_y[indices]
        new_training_x = training_x[:-num_validating_x]
        new_training_y = training_y[:-num_validating_x]
        new_validating_x = training_x[-num_validating_x:]
        new_validating_y = training_y[-num_validating_x:]

        historys.append(
            model.fit(
                new_training_x,
                new_training_y,
                validation_data=(new_validating_x, new_validating_y),
                batch_size=128,
                epochs=100,
                callbacks=[
                    EarlyStopping(
                        monitor="val_f1_score",
                        mode="max",
                        patience=10,
                        verbose=1),
                    ModelCheckpoint(
                        "model_{:d}.h5".format(i),
                        save_best_only=True,
                        save_weights_only=True,
                        monitor="val_f1_score",
                        mode="max",
                        verbose=1),
                ]))

    with open("historys.p", "wb") as history_file:
        pickle.dump([h.history for h in historys], history_file)
