
# Wildfire Prediction using custom CNN v/s Pretrained ResNet50 model

Wildfires inflict massive environmental harm and pose danger to human lives globally.
Prediction of danger of ignition early is possible, enabling anticipatory allocation of
resources and evacuation planning

This project assesses machine-learning methods on
the Kaggle Wildfire Prediction Dataset(https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)

This dataset useses Longitude and Latitude coordinates for
each wildfire spot (> 0.01 acres burned) found on the dataset above extracted satellite images of those areas using MapBox API 

# About Dataset:

This dataset contains satellite images (350x350px) in 2 classes :

● Wildfire : 22710 images

● No wildfire : 20140 images

The data was divided into train, test and validation with these percentages :

● Train : ~70%

● Test : ~15%

● Validation : ~15%

# Methodology

## 1) Importing and Loading Data
Used opendatasets library to directly download Kaggle dataset into my Google colab environment

## 2) Data Cleaning and Preprocessing

● To clean data created a function to remove corrupted data from the dataset so that no problem should occur while training

● This function goes through each image and checks it’s integrity if it fails integrity or loading it will be removed from the dataset

● For preprocessing training data we used
Rescaling, Rotation, Width shift, Hight shift, Shear, Zoom, Horizontal Flip and Reflect

## 3) Custom CNN architecture

● First Convolutional Layer (Conv2D(32, (3,3)))

Looks at the input image (224*224*3) through 32 small 3*3 “windows,”Learn basic features like edges and corners.

● First Pooling Layer (MaxPooling2D((2,2)))

Cuts the image size in half (from 224*224 to 112*112) by picking the strongest pixel in each 2*2 block, 

Helps the model focus on the most important parts and reduces computation.

● Second Convolutional Layer (Conv2D(64, (3,3)))

Applies 64 filters of size 3*3 on the down-sampled image, 

Discovers more complex patterns like simple shapes or textures.

● Second Pooling Layer (MaxPooling2D((2,2)))

Again halves the size (from 112*112 to 56*56),

Keeps the strongest signals, making the model more robust to small changes.

● Third Convolutional Layer (Conv2D(128, (3,3)))

Uses 128 filters to capture even finer details,
Learns high-level features like parts of objects or patterns of flames.

● Third Pooling Layer (MaxPooling2D((2,2)))

Downs samples again (to 28*28),Greatly reduces data size before switching to a normal neural network.

● Flatten Layer (Flatten())

Converts the 3D feature maps (28*28*128) into a long 1D list of numbers,Prepares the data for the fully connected layers.

● Fully Connected Layer (Dense(256, 'relu'))

Connects every number from the flattened list to 256 neurons,
Learn higher-level combinations of the features.

● Dropout Layer (Dropout(0.5))

Randomly “turns off” half of those 256 neurons during training,
Prevents the model from memorizing the training data (overfitting).

● Output Layer (Dense(1, 'sigmoid'))

Has a single neuron that outputs a number between 0 and 1,
Values near 1 mean “wildfire,” near 0 mean “no wildfire.”
 Then used Adam Optimiser and Binary_CrossEntropy Loss Function with a epoch of 10
 Implemented an early callback at precision
## 4) Transfer Learning with Resnet50

● Load a Pretrained ResNet50 model
ResNet50 model trained on imagenet dataset with millions of images

Then remove the top classification layer and give input of 224,224 image size

● Freeze the ResNet50 layers
Stops from retraining of previously trained ResNet layers

● Build a simple new “head” on top
Add custom layers for binary classification with sigmoid function
