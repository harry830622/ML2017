#!/usr/bin/env python3

import numpy as np
import pickle
import sys
import itertools

from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


def plot_confusion_matrix(cm,
                          classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:.2f}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


training_x_file_name = sys.argv[1]
training_y_file_name = sys.argv[2]
model_file_name = sys.argv[3]
dump_file_name = sys.argv[4]
prefix = sys.argv[5]

model = load_model(model_file_name)

with open(dump_file_name, "rb") as dump_file:
    m = pickle.load(dump_file)
    history = m["history"]

training_x = []
with open(training_x_file_name, "rb") as training_x_file:
    training_x = pickle.load(training_x_file)

training_y = []
with open(training_y_file_name, "rb") as training_y_file:
    training_y = pickle.load(training_y_file)

training_x = np.array(
    training_x, dtype=np.float64).reshape((len(training_x), 48, 48, 1))
training_y = np.array(training_y, dtype=np.float64)

training_x /= 255

shuffle_index = np.arange(len(training_x))
np.random.seed(19940622)
np.random.shuffle(shuffle_index)
training_x = training_x[shuffle_index]
training_y = training_y[shuffle_index]

num_validating_x = training_x.shape[0] // 10
validating_x = training_x[:num_validating_x]
validating_y = training_y[:num_validating_x]

model.summary()

plot_model(model, "{}_structure.png".format(prefix))

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('# of epochs')
plt.legend(['train', 'valid'], loc='lower right')
plt.savefig("{}_acc.png".format(prefix))
plt.close()

classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
predicted_y = model.predict_classes(validating_x)
cnf_matrix = confusion_matrix(validating_y, predicted_y)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes)
plt.savefig("{}_confusion_matrix.png".format(prefix))
plt.close()

img_ids = range(200, 210)
num_imgs = len(img_ids)
plt.figure(figsize=(8, num_imgs * 2))
input_img = model.input
for nth, i in enumerate(img_ids):
    y = predicted_y[i]
    target = K.mean(model.output[:, y])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])

    heatmap = np.array(fn([np.reshape(validating_x[i], (1, 48, 48, 1)), 1]))
    heatmap = heatmap[0, 0, :, :, 0]
    heatmap = np.abs(heatmap)
    heatmap = heatmap / np.max(heatmap)

    see = validating_x[i].reshape((48, 48))
    thresh = np.mean(see)
    img = see * 255
    see[np.where(heatmap <= thresh)] = thresh
    see *= 255

    print(nth, classes[y])

    plt.subplot(num_imgs, 3, 1 + 3 * nth)
    plt.imshow(img, cmap="gray")

    plt.subplot(num_imgs, 3, 2 + 3 * nth)
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(num_imgs, 3, 3 + 3 * nth)
    plt.imshow(see, cmap="gray")
    plt.colorbar()
    plt.tight_layout()

plt.savefig("{}_heatmap.png".format(prefix))
plt.close()
