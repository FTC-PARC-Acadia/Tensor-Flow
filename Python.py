from cProfile import label
from distutils.log import info
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import layers
from keras.models import Sequential

ds = tf.keras.datasets.mnist

train_ds, val_ds = ds.load_data()

model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(28, 28, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x=train_ds, validation_data=val_ds , epochs=10)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
                                         
img = tf.keras.utils.load_img("3.png")
image_array = tf.keras.utils.img_to_array(img)
image_array = (tf.expand_dims(image_array, 0))

predictions_single = probability_model.predict(image_array)

print(predictions_single)



