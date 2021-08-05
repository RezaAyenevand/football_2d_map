import cv2
import numpy as np
import tensorflow as tf
import glob
import os


from keras.layers import Input ,Dense,Activation, Conv2D,AveragePooling2D,Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt

#get_ipython().magic(u'matplotlib inline')

def build_model(input_shape):
    x_input = Input(shape=input_shape, name='input')

    x = Conv2D(filters=15, kernel_size=(3, 3), strides=1, padding='valid', name='conv1')(x_input)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=2, strides=2, name='pad1')(x)

    x = Conv2D(filters=6, kernel_size=(2, 2), strides=1, padding='valid', name='conv2')(x_input)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=2, strides=2, name='pad2')(x)

    x = Flatten()(x)

    x = Dense(units=50, name='fc_1')(x)

    x = Activation('relu', name='relu_1')(x)
    # x = Dropout(rate = 0.5)

    x = Dense(units=25, name='fc_2')(x)
    x = Activation('relu', name='relu_2')(x)
    # x = Dropout(rate = 0.5)

    outputs = Dense(units=3, name='softmax', activation='softmax')(x)

    model = Model(inputs=x_input, outputs=outputs)
    model.summary()

    return model




path = 'D:/University/3992/Computer Vision/Project/Sample1/Red'
images = []
labels = []
for filename in glob.glob(os.path.join(path, '*.jpg')):
  I = cv2.imread(filename)
  resized = cv2.resize(I, (70 , 70), interpolation = cv2.INTER_AREA)
  images.append(resized)
  labels.append(0)
path = 'D:/University/3992/Computer Vision/Project/Sample1/Blue'
for filename in glob.glob(os.path.join(path, '*.jpg')):
  I = cv2.imread(filename)
  resized = cv2.resize(I, (70 , 70), interpolation = cv2.INTER_AREA)
  images.append(resized)
  labels.append(1)
path = 'D:/University/3992/Computer Vision/Project/Sample1/Referee'
for filename in glob.glob(os.path.join(path, '*.jpg')):
  I = cv2.imread(filename)
  resized = cv2.resize(I, (70 , 70), interpolation = cv2.INTER_AREA)
  images.append(resized)
  labels.append(2)
img = np.array(images)
labels = np.array(labels)
print(img.shape)
print(labels.shape)

datagen = ImageDataGenerator(
          rotation_range=30,
          width_shift_range=0.2,
          height_shift_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')


x_train, x_test, y_train, y_test  = train_test_split(img, labels , test_size=0.1, shuffle=True, random_state=5)

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
print(x_train.shape)
print(y_train.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("x_train shape is {}".format( x_train.shape ))
print("y_train shape is {}".format( y_train.shape ))
print("x_test shape is {}".format( x_test.shape ))
print("y_test shape is {}".format( y_test.shape ))

model = build_model(input_shape=(70,70,3))
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy' ,metrics = ['accuracy'])


batch_size = 8

H = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        steps_per_epoch=len(y_train) // batch_size, epochs=20)

model.save("./model")