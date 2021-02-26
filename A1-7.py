import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.initializers import TruncatedNormal, Constant
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.optimizers import SGD


# grayscale or rgb
COLOR_MODE = 'grayscale'

# grayscale: 1, rgb: 3
COLOR_CHANNEL = 1

INPUT_IMAGE_SIZE = 512
BATCH_SIZE = 32
EPOCH_NUM = 30
TRAIN_PATH = r"C:\"

# folder name
folder = ["CE-abdomen","CE-brain","CE-chest","CE-neck","CE-pelvis",
          "P-abdomen","P-brain","P-chest","P-neck","P-pelvis"]
CLASS_NUM = len(folder)
print("class : " + str(CLASS_NUM))


# data set
train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    subset="training")
validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    subset="validation")


# architecture
def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        activation='relu',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def dense(units, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units,
        activation='tanh',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def AlexNet():
    model = Sequential()

    # conv1
    model.add(conv2d(96, 13, strides=(4,4), bias_init=0, input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, COLOR_CHANNEL)))
    model.add(MaxPooling2D(pool_size=(7, 7), strides=(2, 2)))
    model.add(BatchNormalization())

    # conv2
    model.add(conv2d(256, 7, bias_init=1))
    model.add(MaxPooling2D(pool_size=(7, 7), strides=(2, 2)))
    model.add(BatchNormalization())

    # conv3~5
    model.add(conv2d(384, 5, bias_init=0))
    model.add(conv2d(384, 5, bias_init=1))
    model.add(conv2d(256, 5, bias_init=1))
    model.add(MaxPooling2D(pool_size=(7, 7), strides=(2, 2)))
    model.add(BatchNormalization())

    # fc
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))

    # output
    model.add(Dense(CLASS_NUM, activation='softmax'))

    model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = AlexNet()

model.summary()


# train
history = model.fit_generator(
    train_generator,
    steps_per_epoch=250, 
    validation_data=validation_generator,
    validation_steps=62,
    epochs=EPOCH_NUM)


score = model.evaluate_generator(validation_generator, steps=62, verbose=0)
print(len(validation_generator))
print('Loss:', score[0])
print('Accuracy:', score[1])


# save
model_arc_json = model.to_json()
open("model_architecture.json", mode='w').write(model_arc_json)
model.save_weights("weights.hdf5")


#accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

