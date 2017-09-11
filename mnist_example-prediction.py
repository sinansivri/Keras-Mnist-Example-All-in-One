
# coding: utf-8

# In[4]:

#Training a model with Keras-Tensorflow Backend



import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os

batch_size = 128
num_classes = 10
epochs = 8

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),            # 1st Conv layer , 64 nodes , 3x3 kernel , image size 28x28 , relu activation
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))     # 2nd Conv layer , 64 nodes , 3x3 kernel , image size 28x28 , relu activation
model.add(MaxPooling2D(pool_size=(2, 2)))            # Resampling Layer image size 14x14
model.add(Dropout(0.25))                             # Generalization Layer randomly shuts down 0.25 of nodes

model.add(Flatten())
model.add(Dense(64, activation='relu'))              # 1st Fully connected layer , 64 nodes ,relu activation
model.add(Dropout(0.5))                              # Generalization Layer randomly shuts down 0.50 of nodes

model.add(Dense(num_classes, activation='softmax'))  # 2nd Fully connected Layer , 10 nodes for 10 classes,
                                                     # softmax for probablity estimation

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_mnist_trained_model.h5'

if not os.path.isdir(save_dir):
    
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print('Saved trained model at %s ' % model_path) # Save example code model


# In[67]:

# Making predictions from already trained model


import numpy as np
from PIL import Image


img = Image.open( '2.png')
grey=img.convert('L')
grey.load()
data = np.asarray( grey, dtype="float32" );

data = np.reshape(data, (1, 28, 28, 1));
    

model.predict(data,batch_size=1, verbose=0)




# In[29]:

# Load a model and prediction



from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('keras_mnist_trained_model.h5')
    
img = Image.open( '2D.jpg')
grey=img.convert('L')                                           # Convert image to greyscale      
grey.load()
data = np.asarray( grey, dtype="float32" );

data = np.reshape(data, (1, 28, 28, 1));                                             



a=model.predict(data) 
b=model.predict_classes(data)

print ("Class Probabilities \n ",a,"\n Prediction:",b)
    
    


# In[2]:

#Tensorboard trial
#Currently i couldn't get TB to work with Keras. Here is the same example script with TB imports




import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard
import os


batch_size = 128
num_classes = 10
epochs = 8

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),             #1st Conv layer , 64 nodes , 3x3 kernel , image size 28x28 , relu activation
                 activation='relu',                  
                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))     # 2nd Conv layer , 64 nodes , 3x3 kernel , image size 28x28 , relu activation
model.add(MaxPooling2D(pool_size=(2, 2)))            # Resampling Layer image size 14x14
model.add(Dropout(0.25))                             # Generalization Layer randomly shuts down 0.25 of nodes

model.add(Flatten())
model.add(Dense(64, activation='relu'))              # 1st Fully connected layer , 64 nodes ,relu activation
model.add(Dropout(0.5))                              # Generalization Layer randomly shuts down 0.50 of nodes

model.add(Dense(num_classes, activation='softmax'))  # 2nd Fully connected Layer , 10 nodes for 10 classes,
                                                     # softmax for probablity estimation

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_mnist_trained_model_tb_trial.h5'

if not os.path.isdir(save_dir):
    
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print('Saved trained model at %s ' % model_path) # Save example code model


# In[ ]:



