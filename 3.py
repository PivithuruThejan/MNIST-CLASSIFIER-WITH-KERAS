import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
# add noise
noise_factor = 0.25
x_train_noisy = x_train + noise_factor*np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor*np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# dataset dimensions
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# reshaping
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_train.shape)
print(x_test.shape)
x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], 28, 28, 1)
x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0], 28, 28, 1)
print(x_train_noisy.shape)
print(x_test_noisy.shape)
# normalizing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train_noisy = x_train_noisy.astype('float32')
x_test_noisy = x_test_noisy.astype('float32')
x_train_noisy /= 255
x_test_noisy /= 255
# noise removal using an auto encoder
auto_encoder = Sequential()
auto_encoder.add(Conv2D(28, kernel_size=(7,7), activation=tf.nn.relu, padding='same', input_shape=(28, 28, 1)))
auto_encoder.add(MaxPooling2D(pool_size=(2, 2)))
auto_encoder.add(Conv2D(14, kernel_size=(7,7), activation=tf.nn.relu, padding='same'))
auto_encoder.add(MaxPooling2D(pool_size=(2, 2)))
auto_encoder.add(Conv2D(14, kernel_size=(7,7), activation=tf.nn.relu, padding='same'))
auto_encoder.add(UpSampling2D())
auto_encoder.add(Conv2D(28, kernel_size=(7,7), activation=tf.nn.relu, padding='same'))
auto_encoder.add(UpSampling2D())
auto_encoder.add(Conv2D(1, kernel_size=(7,7), activation=tf.nn.sigmoid, padding='same'))

auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
# noise removal
X_noise_train, X_noise_valid, X_train, X_valid = train_test_split(x_train_noisy, x_train, test_size=0.5, random_state=42, shuffle=True)

auto_encoder.fit(X_noise_train, X_train,
                 epochs=2,
                 shuffle=True,
                 validation_data=(X_noise_valid, X_valid)
                 )

x_train_noisy = auto_encoder.predict(x_train_noisy)
x_test_noisy = auto_encoder.predict(x_test_noisy)
# model
model = Sequential()
model.add(Conv2D(28, kernel_size=(7,7), activation=tf.nn.relu, input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train_noisy,y=y_train, epochs=10)
# evaluate model
model.evaluate(x_test_noisy, y_test)
