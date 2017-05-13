#!/usr/bin/env python3

import word2vec
import nltk
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from adjustText import adjust_text

import sys

txt = sys.argv[1]

phrase_file_name = "text_phrases.bin"
word2vec.word2phrase(txt, phrase_file_name, verbose=True)

model_file_name = "text.bin"
word2vec.word2vec(phrase_file_name, model_file_name, verbose=True)

model = word2vec.load(model_file_name)

words = []
word_vectors = []
for word in model.vocab:
    words.append(word)
    word_vectors.append(model[word])
words = np.array(words)
word_vectors = np.array(word_vectors)

num_plotted_words = 1000
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
