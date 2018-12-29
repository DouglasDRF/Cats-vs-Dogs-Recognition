#Force CPU (For benchmark purpose only)                                          
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os, gc, random, csv, time
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras import optimizers
                  
##### Constants #####

TRAIN_DIR = 'data/train/'
TEST_DIR = 'data/test/'

ROWS = 150
COLS = 150
CHANNELS = 3

BATCH_SIZE=64

###### Preparing the data #######

original_train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]

train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

original_train_images = train_dogs[:12000] + train_cats[:12000]  
random.shuffle(original_train_images)

train_images = original_train_images[:18000]
validation_images = original_train_images[18000:] 

def set_data(images):
    count = len(images)
    X = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.float32)
    y = np.zeros((count,), dtype=np.float32)
    
    for i, image_file in enumerate(images):
        img = image.load_img(image_file, target_size=(ROWS, COLS))
        X[i] = image.img_to_array(img)
        if 'dog' in image_file:
            y[i] = 1.
        if i % 1000 == 0: print('Processed {} of {}'.format(i, count))
    
    return X, y

X_train, y_train = set_data(train_images)
X_validation, y_validation = set_data(validation_images)
                                       

######## Pre Processing Image #######

train_datagen = image.ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip = True)
validation_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train,batch_size=BATCH_SIZE)
validation_generator = validation_datagen.flow(X_validation, y_validation, batch_size=BATCH_SIZE)

                                                                                                                                                                                     
########## Model ###########
try:
    model = load_model('dogs_and_cats.h5')
except:
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
    
    train_steps = len(train_images) / BATCH_SIZE
    validation_steps = len(validation_images) / BATCH_SIZE
    
    start_time = time.time()
    history = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=125, validation_data=validation_generator, validation_steps=validation_steps, verbose=1)
    end_time = time.time()
    print('TOTAL TIME TO TRAIN MODEL: ' + str((end_time - start_time) / 60) + 'm')

    model.save('dogs_and_cats.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
                
              
########## Cleaning Memory ########## 

del train_generator
del validation_generator
del X_train, y_train
del X_validation, y_validation

gc.collect()


########### Evaluation ############

evaluation_images = train_dogs[12000:12500] + train_cats[12000:12500]
random.shuffle(evaluation_images)

X_evaluation, y_evaluation = set_data(evaluation_images)
X_evaluation /= 255

evaluation = model.evaluate(X_evaluation, y_evaluation)

X_test, _ = set_data(test_images)
X_test /= 255.

predictions = model.predict(X_test)

for i in range(0, 10):
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

    plt.imshow(image.array_to_img(X_test[i]))
    plt.show()
