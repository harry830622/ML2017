#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVR as SVR

import sys
import os
import pickle


def reconstruct(a, eigen_v):
    m = np.mean(a, axis=0)
    projected_a = np.dot(a - m, eigen_v.T)
    reconstructed_a = m + np.dot(projected_a, eigen_v)
    return reconstructed_a


np.random.seed(19940622)

hands_dir = sys.argv[1]
img_file_names = [
    file_name for file_name in os.listdir(hands_dir)
    if file_name.endswith(".png")
]
img_file_names.sort()

hands = []
for img_file_name in img_file_names:
    hand_img = Image.open(os.path.join(hands_dir, img_file_name))
    hand = np.array(hand_img).flatten()
    hands.append(hand)
hands = np.array(hands)

pca = PCA()
pca.fit(hands)

eigenhands = pca.components_

result_d = hands.shape[0]
variance_ratio_sum = 0.0
for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
    variance_ratio_sum += variance_ratio
    if variance_ratio_sum >= 0.9:
        result_d = i + 1
        break

print("Estimated # of dimensions: {:d}".format(result_d))

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.axis("off")
    plt.imshow(eigenhands[i].reshape(512, 480), cmap="gray")
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.suptitle("eigenhands")
plt.savefig("eigenhands.png")
plt.close()

reconstructed_hands = reconstruct(hands, eigenhands[:result_d])

plt.figure()
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.axis("off")
    plt.imshow(reconstructed_hands[i].reshape(512, 480), cmap="gray")
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.suptitle("reconstructed hands")
plt.savefig("reconstructed_hands_{:d}.png".format(result_d))
plt.close()
