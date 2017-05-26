#!/usr/bin/env python3

from configs import num_models, thresh, wordvec_dimension
from extract import extract_testing_texts
from train import build_model

import numpy as np

from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

import sys
import pickle

testing_file_name = sys.argv[1]
output_file_name = sys.argv[2]
word_index_file_name = sys.argv[3]
label_mapping_file_name = sys.argv[4]

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
embedding_layer = Embedding(
    num_words,
    wordvec_dimension,
    input_length=sequence_length,
    trainable=False)

num_classes = len(classes)
model = build_model(sequence_length, num_classes, embedding_layer)
model.summary()

predicted_ys = []
for i in range(num_models):
    model_file_name = "model_{:d}.h5".format(i)
    model.load_weights(model_file_name)

    predicted_ys.append(model.predict(testing_x))
predicted_ys = np.array(predicted_ys)

classes = np.array(classes)

# Mean
predicted_y = np.mean(predicted_ys, axis=0)
with open(output_file_name, "w") as output_file:
    output_file.write("\"id\",\"tags\"\n")
    for i, y in enumerate(predicted_y):
        tags = " ".join(classes[np.argwhere(y > thresh).flatten()])
        if len(tags) == 0:
            tags = classes[np.argmax(y)]
        output_file.write("\"{:d}\",\"{}\"\n".format(i, tags))

# Voting
# predicted_ys = (predicted_ys > thresh).astype(int)
# predicted_y = np.sum(predicted_ys, axis=0)
# with open(output_file_name, "w") as output_file:
#     output_file.write("\"id\",\"tags\"\n")
#     for i, y in enumerate(predicted_y):
#         tags = " ".join(classes[np.argwhere(y > num_models // 2).flatten()])
#         if len(tags) == 0:
#             tags = classes[np.argmax(y)]
#         output_file.write("\"{:d}\",\"{}\"\n".format(i, tags))
