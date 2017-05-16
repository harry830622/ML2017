#!/usr/bin/env python3

import nltk

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import sys
import pickle

raw_training_file_name = sys.argv[1]
raw_testing_file_name = sys.argv[2]
training_file_name = sys.argv[3]
corpus_file_name = sys.argv[4]

all_text = []
raw_training_y = []
classes = []
with open(raw_training_file_name, "r") as raw_training_file:
    nth_row = 0
    for row in raw_training_file:
        nth_row += 1
        if nth_row != 1:
            columns = row.split(",")
            labels = columns[1].strip("\"").split(" ")
            text = " ".join(columns[2:])
            all_text.append(text)
            raw_training_y.append(labels)
            classes += [label for label in labels if not label in classes]

with open(raw_testing_file_name, "r") as raw_testing_file:
    nth_row = 0
    for row in raw_testing_file:
        nth_row += 1
        if nth_row != 1:
            columns = row.split(",")
            text = " ".join(columns[1:])
            all_text.append(text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(all_text)
sequences = pad_sequences(sequences, padding="post")

num_training_sequences = len(raw_training_y)
num_classes = len(classes)  # 38
training_y = [[i for i, c in enumerate(classes) if c in labels]
              for labels in raw_training_y]
training_y = [[1 if j in classes else 0 for j in range(num_classes)]
              for classes in training_y]

with open(training_file_name, "wb") as training_file:
    pickle.dump({
        "training_sequences": sequences[:num_training_sequences],
        "testing_sequences": sequences[num_training_sequences:],
        "word_index": word_index,
        "classes": classes,
        "training_y": training_y
    }, training_file)

corpus = ""
with open(corpus_file_name, "w") as corpus_file:
    for text in all_text:
        corpus += " ".join(nltk.word_tokenize(text))
    corpus_file.write(corpus)
