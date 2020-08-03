######################################################################
#### Script to build and train model for behavioral cloning project####
######################################################################


################Import libaries#################
import os
import csv
import cv2
import numpy as np
from urllib.request import urlretrieve
from zipfile import ZipFile

import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

##Open driving log file and process data
samples = [] #array to store entries driving log file

with open('./data/data/driving_log.csv') as csvfile: #currently after extracting the file is present in this path
    reader = csv.reader(csvfile)
    next(reader, None) #skip first line as its the headings
    for lines in reader: #read each line and append to samples
        samples.append(lines)
        
        
##################Create training and validation sets#############

#split dataset into training (85%) and validation (15%)
training_samples, validation_samples = train_test_split(samples,test_size=0.15) 

#################Code for building generator####################

#Define generator function to geenerate samples to feed into training process
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: #Continues until ended
        shuffle(samples) #shuffle  images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            #Retrieve corresponding images (to driving log) from data set
            for batch_sample in batch_samples: 
                    for i in range(0,3): #3 images: first -> center, second -> left and third -> right
                        name = './data/data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) #CV2 reads img in BGR, so we convert to RGB
                        steering_angle = float(batch_sample[3]) # steering angle
                        images.append(center_image)
                        
                        #Correct the steering angle for left and right cameras
                        if(i==0):
                            angles.append(steering_angle)
                        elif(i==1):
                            angles.append(steering_angle + 0.2) #increase steering angle by 0.2 for left camera
                        elif(i==2):
                            angles.append(steering_angle - 0.2) #decrease steering angle by 0.2 for right camera
                        
                        ###Augment data: flip image and change sign (+ --> - ) of steering angle
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(steering_angle*-1)
                        elif(i==1):
                            angles.append((steering_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((steering_angle-0.2)*-1)
        
            #Defining x and y variables for training
            y_train = np.array(angles)
            X_train = np.array(images)
            
            yield sklearn.utils.shuffle(X_train, y_train) #yield values by holding it until the generator is running
            
# compile and train model using generator function
train_generator = generator(training_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


###################Code to build and train model###################

#Import Keras libaries
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten, Activation, Dropout


#Initialize model
model = Sequential()

# Preprocess data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) #center to zero with small standard deviation 

# Crop image
model.add(Cropping2D(cropping=((70,25),(0,0)))) #Remove top 70 and bottom 25 pixels          

#Layer 1 - Convolution
model.add(Convolution2D(24,5,5,subsample=(2,2))) #24 filters, 5x5 filter size, 2x2 stride
model.add(Activation('elu')) #ELU activation

#Layer 2 - Convolution 
model.add(Convolution2D(36,5,5,subsample=(2,2))) #36 filters, 5x5 filter size, 2x2 stride
model.add(Activation('elu')) #ELU activation

#Layer 3 - Convolution
model.add(Convolution2D(48,5,5,subsample=(2,2))) #48 filters, 5x5 filter size, 2x2 stride
model.add(Activation('elu')) #ELU activation

#Layer 4- Convolution 
model.add(Convolution2D(64,3,3)) #64 filters, 5x5 filter size, 1x1 stride
model.add(Activation('elu')) #ELU activation

#Layer 5 - Convolution 
model.add(Convolution2D(64,3,3)) #64 filters, 3x3 filter size, 1x1 stride
model.add(Activation('elu')) #ELU activation

#Flatten image
model.add(Flatten())

#Layer 6 - Fully connected layer
model.add(Dense(100)) #100 outputs
model.add(Activation('elu'))

#Dropout layer - 25% 
model.add(Dropout(0.25))

#Layer 7 - Fully connected layer
model.add(Dense(50)) #50 outputs
model.add(Activation('elu'))


#layer 8- Fully connected layer
model.add(Dense(10)) #10 outputs
model.add(Activation('elu'))

#layer 9- Fully connected layer
model.add(Dense(1)) #1 output (steering angle)

# Compile model
model.compile(loss='mse',optimizer='adam') #MSE loss function and adam optimizer

#Use fit generator as the # of images are generated by the generator
model.fit_generator(train_generator, samples_per_epoch= len(training_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1) 

#Save model
model.save('model_new.h5')
print('Model Saved')

"""
#Load pre-trained model - to comment out when training model
import h5py
from keras.models import load_model
model = load_model('model.h5')
"""

#Print model summary
model.summary()

            


