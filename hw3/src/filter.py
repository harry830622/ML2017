#!/usr/bin/env python3

from scipy.misc import imsave
import numpy as np
import math
import pickle
import time
import sys

from keras import backend as K
from keras.models import load_model

model_file_name = sys.argv[1]
prefix = sys.argv[2]

model = load_model(model_file_name)

model.summary()

# dimensions of the generated pictures for each filter.
img_width = 48
img_height = 48

# the name of the layer we want to visualize
layer_name = "conv2d_1"

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer = layer_dict[layer_name]
kept_filters = []
for filter_index in range(32):
    print("Processing filter %d" % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer.output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, img_width, img_height, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    for i in range(100):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print("Filter %d processed in %ds" % (filter_index, end_time - start_time))

n = math.floor(len(kept_filters) ** 0.5)

# the filters that have the highest loss are assumed to be better-looking.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 5 x 5 filters of size 46 x 46, with a 5px margin in between
margin = 3
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 1))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i:(img_width + margin) * i +
                         img_width, (img_height + margin) * j:
                         (img_height + margin) * j + img_height, :] = img

# save the result to disk
imsave("{}_stitched_filters_{}.png".format(prefix, layer_name),
       stitched_filters[:, :, 0])
