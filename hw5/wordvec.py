#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import nltk

from sklearn.manifold import TSNE
from adjustText import adjust_text

import sys

wordvec_file_name = sys.argv[1]

words = []
with open(wordvec_file_name, "r") as wordvec_file:
    word_vectors = []
    for line in wordvec_file:
        columns = line.split(" ")
        words.append(columns[0])
        word_vectors.append([float(s) for s in columns[1:]])

words = np.array(words)
word_vectors = np.array(word_vectors)

num_plotted_words = 2000
words = words[:num_plotted_words]
word_vectors = word_vectors[:num_plotted_words]

tsne = TSNE(n_components=2)
reduced_vectors = tsne.fit_transform(word_vectors)

use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]

plt.figure(figsize=(16, 9))
texts = []
for i, label in enumerate(words):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags and
            all(c not in label for c in puncts)):
        x, y = reduced_vectors[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

plt.savefig('tsne.png')
