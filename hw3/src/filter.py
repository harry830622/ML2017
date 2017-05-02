#!/usr/bin/env python3

from scipy.misc import imsave
import numpy as np
import pickle
import time

from keras import backend as K
from keras.models import model_from_json

model_file_name = "./model.p"

with open(model_file_name, "rb") as model_file:
    m = pickle.load(model_file)
    model = model_from_json(m["config"])
    model.set_weights(m["weights"])

model.summary()

# dimensions of the generated pictures for each filter.
img_width = 48
img_height = 48

# the name of the layer we want to visualize
layer_name = "conv2d_4"

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

kept_filters = []
for filter_index in range(128):
    print("Processing filter %d" % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, img_width, img_height, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(120):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print("Current loss value:", loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print("Filter %d processed in %ds" % (filter_index, end_time - start_time))

# we will stich the best 25 filters on a 5 x 5 grid.
n = 1

# the filters that have the highest loss are assumed to be better-looking.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 5 x 5 filters of size 46 x 46, with a 5px margin in between
margin = 5
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
imsave("%s_stitched_filters_%dx%d.png" % (layer_name, n, n),
       stitched_filters[:, :, 0])
