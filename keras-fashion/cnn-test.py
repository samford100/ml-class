import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

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
config.img_width = 299
config.img_height = 299
#config.img_width = 28
#config.img_height = 28
config.epochs = 10

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

# create model
model=Sequential()

model.add(Conv2D(filters=32, strides=(1,1), kernel_size=5, padding='valid'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=32, strides=(1,1), kernel_size=5, padding='valid'))
model.add(MaxPooling2D(2,2))

model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(num_classes, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=labels)])


model.fit_generator(
    train_generator,
    epochs=config.epochs,
    workers=2,
    steps_per_epoch=nb_train_samples * 2 / config.batch_size,
    validation_data=validation_generator,
    validation_steps=nb_train_samples / config.batch_size,
    callbacks=[WandbCallback(data_type="image", generator=validation_generator, labels=['cat', 'dog'],save_model=False)],
    class_weight='auto')


