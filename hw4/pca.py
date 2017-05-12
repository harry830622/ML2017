#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import os


def pca(a):
    m = a - np.mean(a, axis=0)
    _, _, v = np.linalg.svd(m)
    return v


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

eigen_faces = pca(faces)

plt.figure()
for i in range(9):
    plt.subplot(331 + i)
    plt.axis("off")
    plt.imshow(eigen_faces[i].reshape(64, 64), cmap="gray")
plt.savefig("eigen_faces.png")
plt.close()

faces = []
for i in range(10):
    for j in range(10):
        bmp_file_name = "{}{:02d}.bmp".format(chr(ord("A") + i), j)
        face_img = Image.open(os.path.join(faces_dir, bmp_file_name))
        face = np.array(face_img).flatten()
        faces.append(face)
faces = np.array(faces)

num_pcs = 5
reconstructed_faces = reconstruct(faces, eigen_faces[:num_pcs])

plt.figure()
for i in range(100):
    plt.subplot(10, 20, i * 2 + 1)
    plt.axis("off")
    plt.imshow(faces[i].reshape(64, 64), cmap="gray")
    plt.subplot(10, 20, i * 2 + 2)
    plt.axis("off")
    plt.imshow(reconstructed_faces[i].reshape(64, 64), cmap="gray")
plt.savefig("reconstructed_faces_{:d}.png".format(num_pcs))
plt.close()

faces = []
for i in range(10):
    for j in range(10):
        bmp_file_name = "{}{:02d}.bmp".format(chr(ord("A") + i), j)
        face_img = Image.open(os.path.join(faces_dir, bmp_file_name))
        face = np.array(face_img).flatten()
        faces.append(face)
faces = np.array(faces)

for num_pcs in range(1, 101):
    reconstructed_faces = reconstruct(faces, eigen_faces[:num_pcs])

    rmse = (np.mean(((reconstructed_faces - faces) / 255)**2))**0.5
    print("# of pcs: {:d} RMSE: {:4.2f}%".format(num_pcs, rmse * 100))
    if rmse < 0.01:
        break
