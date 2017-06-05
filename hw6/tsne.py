#!/usr/bin/env python3

import mf

from config import NUM_USERS, NUM_MOVIES, LATENT_DIMENSION
from config import IS_NORMALIZED, IS_REGULARIZED, LAMBDA, IS_BIASED
from extract import MOVIE_GENRES, GENDER, AGE
from extract import extract_users, extract_movies

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patch

from sklearn.manifold import TSNE
from adjustText import adjust_text

import os
import sys

users_file_name = sys.argv[1]
movies_file_name = sys.argv[2]
model_file_name = sys.argv[3]

users = extract_users(users_file_name, num_users=NUM_USERS)
movies = extract_movies(movies_file_name, num_movies=NUM_MOVIES)

tsne_file_name = "tsne.npz"

if not os.path.isfile(tsne_file_name):
    model = mf.build(
        num_users=NUM_USERS,
        num_movies=NUM_MOVIES,
        latent_dimension=LATENT_DIMENSION,
        is_regularized=IS_REGULARIZED,
        lamda=LAMBDA,
        is_biased=IS_BIASED)

    model.summary()

    model.load_weights(model_file_name)

    user_latent = model.get_layer(name="UserLatent").get_weights()[0]
    movie_latent = model.get_layer(name="MovieLatent").get_weights()[0]

    tsne = TSNE(n_components=2)
    user_tsne = tsne.fit_transform(user_latent)
    movie_tsne = tsne.fit_transform(movie_latent)

    np.savez(tsne_file_name, user_tsne=user_tsne, movie_tsne=movie_tsne)
else:
    tsne = np.load(tsne_file_name)
    user_tsne = tsne["user_tsne"]
    movie_tsne = tsne["movie_tsne"]

plt.figure(figsize=(16, 9))

movie_by_genre = [[] for i in range(18)]
for i, (m, t) in enumerate(zip(movies, movie_tsne)):
    genres = [g for n, g in zip(m, MOVIE_GENRES) if n == 1]
    idx = 0
    if "Drama" in genres and "Musical" in genres:
        idx = 11
    if "Thriller" in genres and "Action" in genres:
        idx = 15
    movie_by_genre[idx].append((i, t))

draw = [11, 15]

texts = []
colors = cm.rainbow(np.linspace(0, 1, num=len(MOVIE_GENRES)))
for i, (vs, c) in enumerate(zip(movie_by_genre, colors)):
    if i in draw:
        for movie_id, v in vs:
            x, y = v
            plt.scatter(x, y, c=c)
            # texts.append(plt.text(x, y, str(movie_id)))

rects = []
for i, c in enumerate(colors):
    if i in draw:
        rects.append(patch.Rectangle((0, 0), 1, 1, fc=c))

plt.legend(
    rects, [g for i, g in enumerate(MOVIE_GENRES) if i in draw],
    loc="lower right")

# adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

plt.savefig("movie_genre_tsne.png")
plt.close()

plt.figure(figsize=(16, 9))

user_by_gender = [[] for i in range(2)]
for i, (u, t) in enumerate(zip(users, user_tsne)):
    v = u[:2]
    idx = np.where(v == 1)[0]
    if idx.shape[0] != 0:
        user_by_gender[idx[0]].append((i, t))

texts = []
colors = ["b", "r"]
for vs, c in zip(user_by_gender, colors):
    for user_id, v in vs:
        x, y = v
        plt.scatter(x, y, c=c)
        # texts.append(plt.text(x, y, str(user_id)))

rects = []
for c in colors:
    rects.append(patch.Rectangle((0, 0), 1, 1, fc=c))

plt.legend(rects, GENDER, loc="lower right")

plt.savefig("user_gender_tsne.png")
plt.close()

plt.figure(figsize=(16, 9))

user_by_age = [[] for i in range(7)]
for i, (u, t) in enumerate(zip(users, user_tsne)):
    v = u[2:9]
    idx = np.where(v == 1)[0]
    if idx.shape[0] != 0:
        user_by_age[idx[0]].append((i, t))

texts = []
colors = cm.rainbow(np.linspace(0, 1, num=len(AGE)))
for vs, c in zip(user_by_age, colors):
    for user_id, v in vs:
        x, y = v
        plt.scatter(x, y, c=c)
        # texts.append(plt.text(x, y, str(user_id)))

rects = []
for c in colors:
    rects.append(patch.Rectangle((0, 0), 1, 1, fc=c))

plt.legend(rects, AGE, loc="lower right")

plt.savefig("user_age_tsne.png")
plt.close()
