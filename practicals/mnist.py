import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

# load the MNIST the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scale the image intensities to the 0-1 range
x_train = (x_train / 255.0).astype(np.float32)
x_test = (x_test / 255.0).astype(np.float32)

# convert the data to channel-last
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# convert the labels to one-hot encoded
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

def plot_images(images, dim=(10, 10), figsize=(10, 10), title=''):
    
    plt.figure(figsize=figsize)
    
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
    
plot_images(x_train[np.random.randint(0, x_train.shape[0], size=100)].reshape(100, 28, 28))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

# Original model
original_model = Sequential()
original_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
original_model.add(Conv2D(64, (3, 3), activation='relu'))
original_model.add(MaxPooling2D(pool_size=(2, 2)))
original_model.add(Dropout(0.25))
original_model.add(Flatten())
original_model.add(Dense(128, activation='relu'))
original_model.add(Dropout(0.5))
original_model.add(Dense(10, activation='softmax'))

original_model.summary()

# Fully convolutional model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Replace Flatten and Dense layers with Conv2D layers
model.add(Conv2D(128, (7, 7), activation='relu'))  # This layer replaces Flatten and first Dense layer
model.add(Dropout(0.5))
model.add(Flatten())  # Add Flatten layer to reshape the output
model.add(Dense(10, activation='softmax'))  # This layer replaces the second Dense layer

model.summary()

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

original_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
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