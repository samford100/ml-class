import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback
import os
import sys
import glob
import argparse

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

# logging code
run = wandb.init()
config = run.config
config.first_layer_convs = 32
config.first_layer_conv_width = 5
config.first_layer_conv_height = 5
config.second_layer_conv_width = 5
config.second_layer_conv_height = 5
config.dropout = 0.2
config.dense_layer_size = 128
#config.img_width = 28
#config.img_height = 28
config.img_width = 299
config.img_height = 299
config.epochs = 10
config.fc_size = 1024


# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(X_train.shape[0], config.img_width, config.img_height, 1)
X_test = X_test.reshape(X_test.shape[0], config.img_width, config.img_height, 1)

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]
 
model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer

x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(config.fc_size, activation='relu')(x) #new FC layer, random init
predictions = Dense(len(labels), activation='softmax')(x) #new softmax layer
model = Model(inputs=model.input, outputs=predictions)
model._is_graph_network = False

NB_IV3_LAYERS_TO_FREEZE = 172

for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
    layer.trainable = False
for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
    layer.trainable = True
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)
print(X_train.shape)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=10,
                    callbacks=[WandbCallback(data_type="image", labels=labels,save_model=False)])


