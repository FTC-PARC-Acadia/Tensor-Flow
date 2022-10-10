from cProfile import label
from distutils.log import info
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import layers
from keras.models import Sequential

import pathlib
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)

ds = tfds.load('emnist/letters', split='train', shuffle_files=True)

# roses = list(data_dir.glob('roses/*'))
# img = mpimg.imread(str(roses[0]))
# imgplot = plt.imshow(img)
# plt.show()


