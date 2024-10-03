import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale the image intensities to the 0-1 range
x_train = (x_train / 255.0).astype(np.float32)
x_test = (x_test / 255.0).astype(np.float32)

# Convert the data to channel-last format (28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Convert the labels to one-hot encoded format
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Function to plot a grid of images
def plot_images(images, dim=(10, 10), figsize=(10, 10), title=''):
    plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

# Plot 100 random images from the training set
plot_images(x_train[np.random.randint(0, x_train.shape[0], size=100)].reshape(100, 28, 28))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

# Define the original model
original_model = Sequential()
original_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
original_model.add(Conv2D(64, (3, 3), activation='relu'))
original_model.add(MaxPooling2D(pool_size=(2, 2)))
original_model.add(Dropout(0.25))
original_model.add(Flatten())
original_model.add(Dense(128, activation='relu'))
original_model.add(Dropout(0.5))
original_model.add(Dense(10, activation='softmax'))

# Print the summary of the original model
original_model.summary()

# Define the fully convolutional model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Replace Flatten and Dense layers with Conv2D layers
model.add(Conv2D(128, (3, 3), activation='relu'))  # This layer replaces Flatten and first Dense layer
model.add(Dropout(0.5))
model.add(Conv2D(10, (1, 1), activation='relu'))  # This layer replaces the second Dense layer
model.add(Flatten())  # Flatten the output to match the shape for Dense layer
model.add(Dense(10, activation='softmax'))  # Final Dense layer for classification

# Print the summary of the fully convolutional model
model.summary()

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

# Compile the original model
original_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

# Compile the fully convolutional model
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

# Train the original model
original_model.fit(x_train, y_train,
                   batch_size=128,
                   epochs=1,
                   verbose=1,
                   validation_data=(x_test, y_test))

# Train the fully convolutional model
model.fit(x_train, y_train,
          batch_size=128,
          epochs=1,
          verbose=1,
          validation_data=(x_test, y_test))

# Evaluate the original model
original_score = original_model.evaluate(x_test, y_test, verbose=0)
print('Original model - Test loss:', original_score[0])
print('Original model - Test accuracy:', original_score[1])

# Evaluate the fully convolutional model
score = model.evaluate(x_test, y_test, verbose=0)
print('Fully convolutional model - Test loss:', score[0])
print('Fully convolutional model - Test accuracy:', score[1])