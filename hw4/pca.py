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


def reconstruct(a, eigen_v):
    m = np.mean(a, axis=0)
    projected_a = np.dot(a - m, eigen_v.T)
    reconstructed_a = m + np.dot(projected_a, eigen_v)
    return reconstructed_a


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

faces = []
for i in range(4):
    for j in range(25):
        bmp_file_name = "{}{:02d}.bmp".format(chr(ord("A") + i), j)
        face_img = Image.open(os.path.join(faces_dir, bmp_file_name))
        face = np.array(face_img).flatten()
        faces.append(face)
faces = np.array(faces)

reconstructed_faces = reconstruct(faces, eigen_faces[:5])

plt.figure()
for i in range(100):
    plt.subplot(10, 20, i * 2 + 1)
    plt.axis("off")
    plt.imshow(faces[i].reshape(64, 64), cmap="gray")
    plt.subplot(10, 20, i * 2 + 2)
    plt.axis("off")
    plt.imshow(reconstructed_faces[i].reshape(64, 64), cmap="gray")
plt.savefig("reconstructed_faces.png")
plt.close()
