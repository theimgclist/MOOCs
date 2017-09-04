# Convolutional Neural Network
# CNN is convolution applied on top of ANN
# To preserve the structure in images
# mainly used for images or videos
# goal is to classify the image as a cat or a dog
# we will first train CNN with cat and dog images
# then we will make predictions from new data
# the same can be applied for other binary classification
# to see if there is brain tumor or not is one example
# until now we worked on data in table form
# here we have images
# we dont have data like before in IV and DB form
# we divide our data into 2 folders, train and test
# in each of those, we can create a folder for each class
# inside each class, we name all the data with the name of the class
# there is another way to do it, which is using Keras
# Keras does it efficiently
# Keras is used mainly for DL and computer vision
# we create the right folder structure for Keras
# this is a very useful dataset which can be used to evaluate the performance of DL models
# We have 10000 samples in total, 8000 in train set and 2000 in test set
# this takes care of importing dataset, splitting data, there is no encoding here
# we should do feature scaling which is very important in computer vision
# 


# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D # used in the first step, to add convolutional layers 
from keras.layers import MaxPooling2D # step 2 pooling layers
from keras.layers import Flatten # step 3, convert pool feature maps to feature vector
from keras.layers import Dense # add fully connected layers to NN

# Initialising the CNN
# used to initialize our NN
# there are two ways to init a NN
# either using a graph or sequence of layers
#  
classifier = Sequential() # initializing the classifier CNN

# Step 1 - Convolution
# Creating convolution layer
# 32 is number of filters, feature detectors
# the defualt number is 64
# but we start with 32 and add more for other layers
# also we are using CPU, so its better to go with 32
# 32 feature detectors with 3 rows and 3 columns
# we will have images of different sizes
# so we should have a specific format for all which we mention in input_shape
# 3 is for color images, we should use 2 for black and white
# we took 64 here because we are working on CPU
# with GPU we can use 256
# in ANN we used activation function rectifier for activatin neurons in hidden layer
# here we are using it to makesure we dont have any -ve values or pixels 

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# to reduce the size of the feature map
# 2 X 2 is the slide size
# by doing pooling, we still retain the useful information
#
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer(added at the end of tutorial to show how it improves the accuracy)
# in the 2nd CL, we are not adding input_shape
# because in CL1 we are giving new image so we should specify shape
# but for CL2, the input the pooled filter map

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# why will the model not lose data on flattening?
# starting from the image input, we are taking the max value and retaining throughout
# the same high values are stored in flattened vector, so no change in image
# why not take the full input image?
# if we take each pixel of input image, it will be independent and no spatial relationship is retained
# if we take feature detectors and pooling, we are keeping track of each feature found and relationship with other pixels
# 
classifier.add(Flatten())

# Step 4 - Full connection
# dense is used to add a fully connected layer
# in ANN we took average number of input and output layer
# we chose 128, which isnt too small or too big
# its better to pick number which is power of 2
# we are using rectifier since its a hidden layer
# we are using sigmoid for output later with one node

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) # output layer

# Compiling the CNN
# adam is the algorithm that implements stochaistic gradient descent
# since its binary classification, we used binary_crossentropy

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# augumentation = preprocessing the images to prevent overfitting
# if we dont do it, we will get good accuracy with training data but less accuracy in test or new data
# that is what happens with overfitting
# overfitting happens when there is not enough data
# of the available data the model learns some patterens
# which when applied to new data, dont give accurate results
# we have 10000 data samples which isnt much
# for the CNN to work we need more data
# we can either get more data, or use existing data to make more
# imagedatagenerator is kind of a trick to make new data
# by rotating, random transformations like shifting, flipping
# that gives us lot more diverse data
# this is what augumentation means
# image augmentation is a technique to enrich our dataset without adding new data
# using IDG we are specifying transformation
# target size is size of our input images
# feature scaling which is important is done with IDG


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# we got 84% acc for training data, 75% for test data
# though 75% for test data is good, it inst because there is 10% diff with train data
# takes some time to fit the CNN
# to imporve the acc, we can either add more convolution layer, or more fully connected layer, or add both
# with each epoch, the acc gradually grows
# after adding a new CL, the acc of test data improved to 82%


classifier.fit_generator(training_set,
                         samples_per_epoch = 8000, # number of samples in train data
                         nb_epoch = 2,
                         validation_data = test_set,
                         nb_val_samples = 2000) # number of samples in testset