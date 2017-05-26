#!/usr/bin/env python3

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle


def extract_testing_texts(testing_file_name):
    all_text = []
    with open(testing_file_name, "r") as testing_file:
        nth_line = 0
        for line in testing_file:
            nth_line += 1
            if nth_line != 1:
                text = line.partition(",")[2]
                all_text.append(text)
    return all_text


def extract(training_file_name, testing_file_name):
    all_text = []

    raw_training_y = []
    classes = []
    with open(training_file_name, "r") as training_file:
        nth_line = 0
        for line in training_file:
            nth_line += 1
            if nth_line != 1:
                labels, _, text = line.partition(",")[2].partition(",")
                labels = labels.strip("\"").split(" ")
                all_text.append(text)
                raw_training_y.append(labels)
                classes += [label for label in labels if label not in classes]

    all_text += extract_testing_texts(testing_file_name)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_text)

    num_training_x = len(raw_training_y)
    all_sequences = tokenizer.texts_to_sequences(all_text)
    all_sequences = pad_sequences(all_sequences)
    training_x = all_sequences[:num_training_x]
    testing_x = all_sequences[num_training_x:]

    num_classes = len(classes)  # 38
    training_y = [[i for i, c in enumerate(classes) if c in labels]
                  for labels in raw_training_y]
    training_y = [[1 if i in label_indices else 0 for i in range(num_classes)]
                  for label_indices in training_y]

    with open("word_index.p", "wb") as word_index_file:
        pickle.dump({
            "tokenizer": tokenizer,
            "sequence_length": len(training_x[0])
        }, word_index_file)

    with open("label_mapping.p", "wb") as label_mapping_file:
        pickle.dump(classes, label_mapping_file)

    return training_x, training_y, tokenizer, testing_x, classes
