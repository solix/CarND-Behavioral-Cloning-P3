# **Behavioral Cloning** 


**Behavioral Cloning Project report by soheil jahanshahi**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/val.png "value"
[image3]: ./examples/data.png "dataset"
[image4]: ./examples/left.jpg "left Image"
[image5]: ./examples/right.jpg "right Image"
[image6]: ./examples/center.jpg "center Image"


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup.md`  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

and to train the model by yourself you can execute command like this
```sh 
python model.py --epochs 12  --learning_rate 0.0001
```
#### 3. Submission code is usable and readable
The `model.py` contains file for data preparation, preprocessing and training the model. It has comments where necessary and the code is formatted so that is readable easily.

### Model Architecture and Training Strategy
The `model.py` file contains the code for generating data in realtime between line  `12` through `85`. Function `generator_batch` load and augment data in batches so to avoid storing the data in to memory.
Also during data generation to I select random index from dataset list to avoid overfitting.

The model is constructed with  couple of convolution layers(explain in detail in next section) with kernel size 3x3 and 5x5 , Before I feed in the data to hidden layer normalisation and cropping is used using `Keras.Lambda` and `Keras.Cropping2D` to normalise the data and crop uninterested objects. After feeding the data through convolutional layers, we feed forward to fully connected layers.

For loss function I have used `mse` and optimiser is `adam` optimiser accepting a learning rate value.

In short training strategy was as follows:

* I first make sure that my data is ready and preprocess to feed in network similar to Nvidia
* I feed in data and trained with multiple epochs to see if loss decreases which it was the case
* I tested the trained model on validation data
* I tested the model on simulator  

Obviously above strategy was not achieved at once, I made various experimentation tuning parameters and add/remove layer to get best possible result, I try to explain my strategies as we go through the report.

Last note before we dive into the model is that I have used data provided by Udacity and modified steering angles for left and right images with offset value of `-/+ 0.2`. 

#### 1. An appropriate model architecture has been employed

You can find the model architecture in `model.py` lines `126`-`170`.

The model is inspired by [Nvidia network architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

My model uses 5 convolutional layers followed by 5 fully connected layers. I have used kernel sizes of `3x3` and `5x5` with `valid` padding.   

The model includes `ELU` layers to introduce nonlinearity (code line 20), and the data is normalised in the model using a Keras lambda layer (for example ta a look at code in line 133). I chose `ELU` after reading [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289) paper. In the paper it claimed that `ELU` will speeds up learning in deep neural networks and leads to higher classification accuracies, this activation function also give a better non-linear output activation for the next layer in the network to train.

#### 2. Attempts to reduce overfitting in the model
 To avoid overfitting I first shuffle the dataset so that orders changes , in `generator_batch` line `80` I choose a random image and append it to the dataset that is batched. but this is not enough for avoiding overfitting. The model contains dropout with value `0.20` for all fully connected layers except the one last in order to reduce overfitting (`model.py` lines `156` to `165`). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimiser, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of centre lane driving, recovering from the left and right sides of the road. I load the dataset path images in to an array using `load_dataset` function and then used `load_and_augment_image` to add data for centre, left and right camera with their offset for their corresponding steering angle. For offset I choose a bias value of `0.2`,`0`,`-0.2` for left, center and right images respectively, that will give us even more data to train.

Before model is fed into the network, couple of preprocessing steps has been taken to ensure that data is balanced. Function `remove_unwanted_data_with_bad_angels` (sorry for the typo for the word angels ;)) in line `12` in `model.py` removes unwanted data which has bad angles. I mean by bad angels here that there are either a angle with values higher or lower than `.80` or is close to `0` with tolerance absolute value of `0.001` (I used `np.math.isclose` from numpy library for removing zero angles with specific tolerance rate).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a model to predict steering angles for car to drive in simulator. Using deep learning models which are already indicated a good result was a good starting point.

I started by coding the whole structure of the model, defined helper functions to load data.

I then started by doing a little bit research to see if I can use already existing models that are result proven. luckily as many suggested `NVIDIA` model mentioned earlier was a good candidate. based on this model I builded up my model.

I applied normalisation to have zero-centred mean for each data and I used cropping inside the model removing down part of image make it easier for the network to look for interesting features.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I then applied generator to create batch features and batch labels for both training and validation data.

To combat the overfitting, I modified the model so that on each densed layer it used `dropout =0.2` as one of the regularisation techniques for neural network. 

Then as a last layer I use a Dense layer of `1` to output the predicted the result.

I used generator and `keras.model.fit_generator` function to feed and load dataset randomly in realtime. I set number of predefined flags for adjusting learning rate, number of epochs and batch size (you can find them in line 118-120 of `model.py` ) 

It was lots of experimenting till I came up with appropriate model and preprocessing.  
 
The final step was to run the simulator to see how well the car was driving around track one. when I trained first version of model with only centred data , car was doing ok until it arrived at the bridge.  there the car was confused and hit the side block. I then added left and right camera and the car drove the bridge successfully but guess what? again car when offtrack on the road where right side was not marked. After some exploration inside dataset I realised that many data in the set has lots of zero angles for steering which gives tendency to the model to predict 0 as angle, so the data was not balanced equally across all number of classes. I could balance that out with `remove_unwanted_data_with_bad_angels` function. I was suprised when I saw the huge improvement, made me think of data preparation is one of super important steps when building a training pipeline for neural networks.   

 At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road with maximum speed of 12.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes. Here is a visualization of the architecture.
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I first started by gathering my own data , but that ended up being a bad data, since I could not capture smooth turns (just a bad gamer) and model did not like it at all during training(giving less accurate predictions), then I switched to Udacity dataset thats when things started to work.
  
here are some images of my recorded images (that I ended up not using):

![alt text][image4]
![alt text][image5]
![alt text][image6]


After the data collection process, and loading which I Explained earlier. I finally randomly shuffled the data set and put 20% of the data into a validation set, in the end i had:

 ![alt text][image3]


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 12 as evidenced by decrease in loss function as number of epochs was increasing for both train and validation set up to that number. training result look like this: 
 ![alt text][image2]


#### 4. Video Result

Here's a [video link to result](./run1.mp4) created using `video.py` file.

<iframe width="350" height="315" src="./run1.mp4" frameborder="0" ></iframe>




#### 5. Improvemenets

The car successfully can drive on track 1 but fails to drive on challange track, in case I have more time to work on this project, these are follwing recommandations I will suggest to implement:

* gather more data from challange track to generalise model more to various road types.
* preprocess data by giving noise, random brightness and maybe trying different color space(I use RGB for now) this will give more interesting input and generalize model as well
* make model drive with higher speed by adding alot of recovery data(for example one track recording from off road and moving back to the center)
