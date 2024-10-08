# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:40:21 2024

@author: amadh
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:56:46 2024

@author: amadh
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224,224]

train_path = 'Datasets/Train'
valid_path = 'Datasets/Test'

#add preprocessing layer to the front of VGG
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights='imagenet',include_top=False)
# We are adding 3 cuz we want 3 channels RGB, imagenet is keras initialized weights, False cuz we don't want the last layer

for layer in vgg.layers:
    layer.trainable = False
    
#The glob module in Python is used to find all the pathnames matching a 
#specified pattern, according to the rules used by Unix shell-style 
#wildcards. It is especially useful for working with files and directories.
#The module provides a convenient way to list files and directories in a 
#directory based on a pattern.

folders = glob('Datasets/Train/*')    

x = Flatten().(vgg.output)
# x is the output of the previous layer which is being passed as input to the dense layer
prediction = Dense(len(folders),activation = 'softmax')(x)
#creating a model object
model = Model(inputs = vgg.input,outputs=prediction)
# view the model structure
model.summary()

# tell the model what cost and optimization method to use
model.compile(loss = 'categorical_corssentropy',
opotimizer = 'adam',
metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Datasets/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Datasets/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)                                       class_mode = 'categorical')

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('facefeatures_new_model.h5')

