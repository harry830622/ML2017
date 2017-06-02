#!/usr/bin/env python3

import mf

from config import NUM_USERS, NUM_MOVIES, LATENT_DIMENSION
from config import IS_NORMALIZED, IS_REGULARIZED, LAMBDA, IS_BIASED
from extract import extract_users, extract_movies

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patch

from sklearn.manifold import TSNE

import sys

movies_file_name = sys.argv[1]
model_file_name = sys.argv[2]

movies = extract_movies(movies_file_name, num_movies=NUM_MOVIES)

model = mf.build(
    num_users=NUM_USERS,
    num_movies=NUM_MOVIES,
    latent_dimension=LATENT_DIMENSION,
    is_regularized=IS_REGULARIZED,
    lamda=LAMBDA,
    is_biased=IS_BIASED)

model.summary()

model.load_weights(model_file_name)

movie_latent = model.get_layer(name="embedding_2").get_weights()[0]

tsne = TSNE(n_components=2)
movie_tsne = tsne.fit_transform(movie_latent)

movie_genres = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

movie_by_genre = [[] for i in range(18)]
for m, t in zip(movies, movie_tsne):
    genres = [g for n, g in zip(m, movie_genres) if n == 1]
    idx = 0
    if "Musical" in genres:
        idx = 11
    if "Thriller" in genres:
        idx = 15
    if "Documentary" in genres:
        idx = 6
    if "Romance" in genres:
        idx = 13
    movie_by_genre[idx].append(t)

colors = cm.rainbow(np.linspace(0, 1, num=len(movie_genres)))

draw = [6, 11, 13, 15]

plt.figure(figsize=(16, 9))
for i, (vs, c) in enumerate(zip(movie_by_genre, colors)):
    if i in draw:
        for v in vs:
            x, y = v
            plt.scatter(x, y, c=c)

rects = []
for i, c in enumerate(colors):
    if i in draw:
        rects.append(patch.Rectangle((0, 0), 1, 1, fc=c))

plt.legend(
    rects, [g for i, g in enumerate(movie_genres) if i in draw],
    loc="lower right")
plt.savefig('tsne.png')
plt.close()
