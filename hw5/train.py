#!/usr/bin/env python3

from metrics import precision, recall, fmeasure

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, GRU, Dense

import sys
import pickle

training_file_name = sys.argv[1]
wordvec_file_name = sys.argv[2]

with open(training_file_name, "rb") as training_file:
    training_data = pickle.load(training_file)

wordvec = {}
with open(wordvec_file_name, "r") as wordvec_file:
    for row in wordvec_file:
        columns = row.split(" ")
        wordvec[columns[0]] = [float(n) for n in columns[1:]]

training_x = np.array(training_data["training_sequences"])
training_y = np.array(training_data["training_y"])
classes = np.array(training_data["classes"])
word_index = training_data["word_index"]

sequence_length = training_x.shape[1]
num_classes = training_y.shape[1]
wordvec_dimension = len(wordvec["the"])

indices = np.arange(training_x.shape[0])
np.random.seed(19940622)
np.random.shuffle(indices)
training_x = training_x[indices]
training_y = training_y[indices]
num_validating_x = training_x.shape[0] // 10

validating_x = training_x[-num_validating_x:]
validating_y = training_y[-num_validating_x:]
training_x = training_x[:-num_validating_x]
training_y = training_y[:-num_validating_x]

num_words = len(word_index)
embedding_matrix = np.zeros((num_words, wordvec_dimension))
for word, i in word_index.items():
    if word in wordvec.keys():
        embedding_matrix[i] = wordvec[word]
    if word.capitalize() in wordvec.keys():
        embedding_matrix[i] = wordvec[word.capitalize()]
    if word.upper() in wordvec.keys():
        embedding_matrix[i] = wordvec[word.upper()]
    if word.lower() in wordvec.keys():
        embedding_matrix[i] = wordvec[word.lower()]

embedding_layer = Embedding(
    num_words,
    wordvec_dimension,
    input_length=sequence_length,
    weights=[embedding_matrix],
    trainable=False)

sequence_input = Input(shape=(sequence_length, ))
embedded_sequence = embedding_layer(sequence_input)
x = LSTM(32, return_sequences=False)(embedded_sequence)
x = Dense(64, activation="relu")(x)
prediction = Dense(num_classes, activation="sigmoid")(x)

model = Model(sequence_input, prediction)

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", precision, recall, fmeasure])

model.fit(
    training_x,
    training_y,
    batch_size=128,
    epochs=3,
    validation_data=(validating_x, validating_y))

testing_x = np.array(training_data["testing_sequences"])
predicted_y = model.predict(testing_x)

with open("predicted.csv", "w") as predicted_csv:
    predicted_csv.write("\"id\",\"tags\"\n")
    i = 0
    for y in predicted_y:
        print(y)
        print(classes[np.argwhere(y > 0.1).flatten()])
        tags = " ".join(classes[np.argwhere(y > 0.1)].flatten())
        predicted_csv.write("{:d},\"{}\"\n".format(i, tags))
        i += 1
