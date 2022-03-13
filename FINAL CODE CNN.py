# Import all necesary libraries
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
#These modules was loaded to buld the Convolutional Neural Network,to make filters and maxpooling, compile, etc.
 
# Load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
#We use the Mnist Data from Keras ONLY to train our CNN.
 
# Scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
#In this step, we setting image format to be preccesed
 
# Define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.05, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 #To our model We use 3 convolutional layers with 32, 64 and 64 filters, and two Maxpooling (2 , 2).
 #We was trying to increase a numer of filters in the last colvolutional layer to 128 but It have been taken 7 hours and 
 #It doesnt finish yet 
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# fit model
	model.fit(trainX, trainY, epochs=10, batch_size=16, verbose=0)
	# save model
	model.save('final_model.h5')
#Prepare to run the model and set the batch size and epochs that is a times to make a interation
 
# entry point, run the test harness
run_test_harness()


#IN THIS PART OF THE CODE WE PREPARE TO LOAD THE TEST INFORMATION AND MAKE THE TEST 
# Make a prediction for a new image.
from numpy import argmax
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model #We load the model obtain in the last step
#These are other libraries to use in this section
 
# Load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0 #Normalize to be from 0 to 1
	return img
 
# load an image and predict the class
def run_example(x):
	# load the image
	img = load_image(x)
	# load model
	model = load_model('final_model.h5')
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	return(digit)
 
# Entry point, run the example
from os import listdir
from os.path import isfile, join
lista=[]
#Here We put the route on our folder where we have Test Images
def ls(ruta = r'C:\Route on your computer'):
    return [arch for arch in listdir(ruta) if isfile(join(ruta, arch))]
#Create a list to read each one of all images
a = ls()
print(run_example(a[9999]))
#Run the test to 10 000 images.
for i in range(10000):
    lista.append(run_example(a[i]))
numeracion = range(0,10000,1)

#Module to export information to CSV format
import pandas as pd
data = {'id':numeracion,
        'Category':lista ,
        }
df = pd.DataFrame(data, columns = ['id', 'Category'])
df.to_csv('averque.csv')

#FINISH
    
    