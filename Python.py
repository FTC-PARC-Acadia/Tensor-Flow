from cProfile import label
from cmath import inf
from distutils.log import info
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

from tensorflow import keras
from keras import layers
from keras.models import Sequential

ds = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = ds.load_data()

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

img = tf.keras.preprocessing.image.load_img("3.png")

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.image.rgb_to_grayscale(img_array)
img_array = np.array([img_array])
img_array = cv2.bitwise_not(img_array)

plt.imshow(img_array[0])
plt.show()

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(np.argmax(score))
print(np.max(score))

