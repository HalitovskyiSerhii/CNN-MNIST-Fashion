from __future__ import absolute_import, division, print_function
import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

#Download MNIST-Fashion data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Normalize data and reshape images 
train_images = train_images / 255.0

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))

test_images = test_images / 255.0

test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

print(train_images.shape)

#Create structure with Keras Sequential API
model = keras.Sequential([

    keras.layers.InputLayer(input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(input_shape=(28, 28, 1),
                        filters=32,
                        kernel_size=4,
                        padding="same",
                        activation=tf.nn.relu),

    keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),

    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    keras.layers.Conv2D(filters=64,
                        kernel_size=[4, 4],
                        padding="same",
                        activation=tf.nn.relu),

    keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),


    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    keras.layers.Flatten(),
    keras.layers.Dense(128,
                       activation=keras.activations.relu,
                       kernel_initializer='he_uniform',
                       bias_initializer='he_uniform'),
    keras.layers.Dense(10, activation=keras.activations.softmax)

])

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    # Interrupt training if `val_acc` stops improving for over 5 epochs
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_acc', mode='max'),
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]


#Separete data on validation and training samples
val_images = train_images[:15000]
val_labels = train_labels[:15000]

partial_train_images = train_images[15000:]
partical_train_labels = train_labels[15000:]

file_name = str(__file__)[:-3]
weights_file = os.path.join('data', 'weights', file_name)
print(weights_file)

#Train model if file with weights not exist, otherwise load them and run model on test data
if not os.path.exists(weights_file):

    open(os.path.abspath(weights_file), 'w').close()

    history = model.fit(partial_train_images, partical_train_labels, epochs=30,
                        batch_size=64, validation_data=(val_images, val_labels), verbose=1,
                        callbacks=callbacks)

    model.save_weights(str(weights_file))
else:
    model.load_weights(str(weights_file))

results = model.evaluate(test_images, test_labels)
print(results)

#Get predictions on test data
predictions = model.predict(test_images)
print(predictions.shape)

predictions1D = np.zeros(10000,dtype=np.int16)

#Reshape predictions maxtrix from [N, 10] to [N, 1], where 10 is number of classes
for row_i in range(len(predictions)):
    predictions1D[row_i] = np.argmax(predictions[row_i])


print(predictions1D)
conf_matrix = np.zeros((10,10),dtype=np.int16)

#Create confusion matrix
for i in range(len(predictions1D)):

    p_class = predictions1D[i]
    t_class = test_labels[i]
    conf_matrix[t_class][p_class] = conf_matrix[t_class][p_class] + 1
        
#Configure plot for more information visualization
      
import itertools

thresh = conf_matrix.max() / 2.0
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, "{:,}".format(conf_matrix[i, j]),
            horizontalalignment="center",
            color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True classes')
plt.xlabel('Predicted classes')
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.title('Confusion matrix')
plt.tight_layout()
plt.show()

