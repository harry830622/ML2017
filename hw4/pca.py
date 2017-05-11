#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import os


def pca(a, num_of_pcs):
    m = a - np.mean(a, axis=0)
    u, s, v = np.linalg.svd(m)
    scores = np.dot(u[:, :num_of_pcs], np.diag(s)[:num_of_pcs, :num_of_pcs])
    pcs = v[:num_of_pcs, :]
    return scores, pcs


faces_dir = os.path.join(os.path.expanduser("~"), "Downloads/face")
bmp_file_names = [
    file_name for file_name in os.listdir(faces_dir)
    if file_name.endswith(".bmp")
]
bmp_file_names.sort()

faces = []
for i in range(10):
    for j in range(10):
        bmp_file_name = "{}{:02d}.bmp".format(chr(ord("A") + i), j)
        face_img = Image.open(os.path.join(faces_dir, bmp_file_name))
        face = np.array(face_img).flatten()
        faces.append(face)
faces = np.array(faces)

_, eigen_faces = pca(faces, 10)

plt.figure()
for i in range(9):
    plt.subplot(331 + i)
    plt.axis("off")
    plt.imshow(eigen_faces[i].reshape(64, 64), cmap="gray")
plt.savefig("eigen_faces.png")
plt.close()

plt.figure()
face_img = Image.open(os.path.join(faces_dir, "A10.bmp"))
face = np.array(face_img).flatten()
plt.axis("off")
plt.imshow(face.reshape(64, 64), cmap="gray")
plt.savefig("reconstructed_faces.png")
plt.close()
