from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Softmax, Conv2D, MaxPooling2D
from keras.utils import np_utils

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# normalize data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print(X_train.shape)

X_train = X_train.reshape((60000,28,28,1))
X_test = X_test.reshape((10000,28,28,1))


print(X_train.shape)
#print(y_train.shape)
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

print('y_train: {}'.format(y_train.shape))

#y_train = y_train.reshape((60000,1,1,10))
#y_test = y_test.reshape((10000,1,1,10))
#print('y_train: {}'.format(y_train.shape))


labels = range(10)


# create model
model=Sequential() # one string of layers -- no divergence or anything
#model.add(Flatten(input_shape=(img_width,img_height))) # (28,28)->(784) # only need to tell the input shape to the first layer in Keras
model.add(Conv2D(filters=6, kernel_size=(3,3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Conv2D(filters=12, kernel_size=(3,3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Flatten()) # (28,28)->(784) # only need to tell the input shape to the first layer in Keras
#model.add(Dense(num_classes, activation='sigmoid')) # Fully connected layer (784)->(10)
model.add(Dense(units=100, activation='relu')) # Fully connected layer (784)->(10)
model.add(Dropout(.4))
model.add(Dense(units=num_classes, activation='softmax')) # Fully connected layer (784)->(10)

model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy']) # metrics -- just says what the model will print as it trains

# Fit the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=labels, save_model=False)])

# print(model.predict(X_test[:10]))
# model.save("model.h5") easy way to save the model