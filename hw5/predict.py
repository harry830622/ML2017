#!/usr/bin/env python3

from extract import extract_testing_texts
from train import build_model

import numpy as np

from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

import sys
import pickle

testing_file_name = sys.argv[1]
output_file_name = sys.argv[2]
model_file_name = sys.argv[3]
word_index_file_name = sys.argv[4]
label_mapping_file_name = sys.argv[5]

with open(word_index_file_name, "rb") as word_index_file:
    d = pickle.load(word_index_file)
    tokenizer = d["tokenizer"]
    sequence_length = d["sequence_length"]

with open(label_mapping_file_name, "rb") as label_mapping_file:
    classes = pickle.load(label_mapping_file)

testing_texts = extract_testing_texts(testing_file_name)
testing_x = tokenizer.texts_to_sequences(testing_texts)
testing_x = pad_sequences(testing_x, maxlen=sequence_length)
testing_x = np.array(testing_x)

num_words = len(tokenizer.word_index) + 1
wordvec_dimension = 300  # glove
embedding_layer = Embedding(
    num_words,
    wordvec_dimension,
    input_length=sequence_length,
    trainable=False)

num_classes = len(classes)
model = build_model(sequence_length, num_classes, embedding_layer)
model.load_weights(model_file_name)
model.summary()

predicted_y = model.predict(testing_x)

classes = np.array(classes)
with open("predicted.csv", "w") as predicted_csv:
    predicted_csv.write("\"id\",\"tags\"\n")
    i = 0
    thresh = 0.4
    for y in predicted_y:
        tags = " ".join(classes[np.argwhere(y > thresh).flatten()])
        if len(tags) == 0:
            tags = classes[np.argmax(y)]
        predicted_csv.write("\"{:d}\",\"{}\"\n".format(i, tags))
        i += 1
