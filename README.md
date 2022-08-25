# Informationsfabrik Tetris Hacksession

## Overview

This repo is the basis for our hacksession on Gamified Image Machine Learning.
You will learn the basics of applying machine learning to image-based problems.
In this repo you will use image-based methods for controlling the classic game Tetris.
We go through multiple steps, starting with a simple color detection through a ML-model that has been trained
externally and is integrated into our program to create your own ML-model using Tensorflow.

## Setup

```sh
# Check out this repository
git clone <url>
cd src

# Install dependencies
pip install --user pipenv
python -m pipenv install --dev
```

## Usage

```sh
# Start the program and follow the instructions
python src/main.py
```

## Workshop stage

### Stage 0: Preparations
Before we get started we'll need to make sure that your camera setup works.
Close all programs that might use your camera (e.g. Microsoft Teams or Zoom) since usually only one program is allowed to use the camera at a time.
Check that your camera works by using your operating system's camera app.

If you see something: Good job!

The program works best if your background is more or less steady and homogenous.
If you sit in front of a window or mirror you might have to move your camera a bit since the reflections and highlights in the window might confuse the program.
For the first part it might also be easier if you can step to the side a bit and hold the post its into the camera without youself being in it, but you can try that out lateron.

Remember to close the camera app again before continuing.

### Stage 1: Color detection

#### Setup

In the first step, we will control the game using a simple color detection.
For this you need 3 post-its (or other items) that have colors which stand out from the background (e.g. blue, pink, yellow).
Start the program and select >Color detection Tetris<.
You will be prompted to configure the colors for your specific camera.
Do that and a new window will open: You should see your webcam image together with the currently applied mask six sliders:
L/U H/S/V. Your task is to adjust the sliders such that the one of the three colors are isolated on the image.

-   Move all L-sliders to the very left and all U-sliders to the very right.
-   Start by holding up a post-it into the camera and raising the L (for lower bound) sliders up until the post-it disappears.
-   Set it to the highest value in which the post-it is still reliably visible.
-   Repeat this for all three "L"-sliders.
-   Continue when with the "U" (for upper bound) and lower it with the same principle.
-   Finally, you should arrive at a setting in which only the post-it is visible and the rest is masked out.
-  When you are satisfied with one color move on and the program will continue with the next.

The six HSV values will be stored in data/color_config.yaml for each post-it color.

#### Usage

Now proceed with the game.
A window should show up which shows your webcam image overlaid with a Tetris game.
You control the game by holding the post-its into the camera.

Have fun!

#### Background

The H, S and V of the sliders stand for Hue, Saturation and Value respectively.
They are a way of representing color ranges.
In computers, we need a numeric way of representing images such that the display hardware can accurately show them.
Most displays use three "sub-pixels" for each pixel in the image in the color red, green and blue whose brightness can be regulated separately.
A usual representation is the "RGB-color space", which means that for each pixel you have 3 integers, often between 0 and 255, to represent the color of a single pixel.

The following pictures show how the RGB and HSV color spaces relate to each other.

![HSV cylinder](/docs/1200px-HSV_color_solid_cylinder_saturation_gray.png)
from https://en.wikipedia.org/wiki/HSL_and_HSV by SharkD
![RGB cube](/docs/RGB_color_solid_cube.png)
from https://en.wikipedia.org/wiki/RGB_color_model by SharkD

In Python, a usual way to represent multi-dimensional matrices like an image often Numpy is used.
Open a terminal and execute this code:

```python
import cv2
cam = cv2.VideoCapture(0)
_, frame = cam.read()
cam.release()
```

Inspect the frame object:

```python
>>> type(frame)
<class 'numpy.ndarray'>
>>> frame.shape
(480, 640, 3)
>>> frame.dtype
dtype('uint8')
```

You get an array with three dimensions: height (in my case 480 pixels), width (640) and number of colors (3) of type uint8, which means that the single frame consists of 480x640x3 unsigned 8 bit integers in total.

Let's look at one specific pixel:

```python
>>> frame[123,456,:]
array([226, 211, 165], dtype=uint8)
```

The default color space for CV2 (the library which we use to extract the frame) is BGR which stands for blue, green, red, which means that the given pixel has a blue value of 226, green 211 and red of 165.

If you want to see how the BGR and HSV channels of your frame look like you can use this snippet:

```python
# show one image per BGR color channel
for (name, chan) in zip(("B", "G", "R"), cv2.split(frame)):
	cv2.imshow(name, chan)

# convert the image to the HSV color space and show it
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)

# loop over each of the individual channels and display them
for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv)):
	cv2.imshow(name, chan)

# wait for a keypress, then close all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```

If you want to get even deeper into the whole theory on color, here is a nice intro: https://www.youtube.com/watch?v=FTKP0Y9MVus

### Stage 2: Teachable Machine Model
Now let's get done to business: in the previous stage we learned about color spaced and numerical representations, but have yet to do some of that machine learning we were lured into this with.

We will now create a model that can differentiate between images based on their contents and not just based on colors.

Open your browser and go to https://teachablemachine.withgoogle.com/train/image

#### Setup

You will arrive at an interface that guides you through the process of creating a simple image model.
* Start by creating the training data:
  * Rename "Class 1" to "Left" and click on "Webcam". You will see webcam image.
  * Think of a pose that you want to mean "Left". This could e.g. be your thumb pointing to the left, or an object like your phone. Be creative!
  * Hold the "Hold the record"-button: Now the program starts collecting your frames. Move around a bit by tilting your hand/object and moving it around your webcam image without leaving it.
  * About 50-100 images should suffice for now.
  * Repeat the process with Class 2, now naming it "Rotate".
  * Add a third class and name it "Right" and repeat.
  * Add a fourh class and name it "Other". Here you should just record the background (or yourself) without any particular action.
* When you are happy with your data click on "Training". Your model is now trained on your machine, the data is not sent to Google.
* Export the model, download it as as Tensorflow Keras model ("Tensorflow" tab, select "Keras" and press on "Download my model"). You should receive a zip-file with two files: Your model and a list of your class names. Place both files (unzipped) in `data\models\teachablemachine`

#### Usage
Run `python src/main.py` and select Teachable Machine as mode.

Have fun!

#### Background
What you just did is that you created your (possibly first) machine learning model!
Let's look what it is:

```python
import tensorflow as tf

# load the model
model = tf.keras.models.load_model("data/models/teachablemachine/keras_model.h5")
model.summary()
model.input_shape
model.output_shape
```

should yield something like this:

```
>>> model.summary()
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================

 sequential_3 (Sequential)   (None, 3)                 128400

=================================================================
Total params: 538,608
Trainable params: 524,528
Non-trainable params: 14,080
_________________________________________________________________
>>> model.input
<KerasTensor: shape=(None, 224, 224, 3) dtype=float32 (created by layer 'sequential_1_input')>
>>> model.output
<KerasTensor: shape=(None, 3) dtype=float32 (created by layer 'sequential_3')>
```

This shows you that you have a model with around half a million parameters, which reads in data of shape (None, 224, 224, 3) and outputs data of shape (None, 3).
Why the Nones?
The Nones represent the batch-dimension. Your model can handle multiple images in parallel and it does not know how many we want to have in parallel at this point, so it shows up as None.
Ignore it for now.
The rest of the input shape are again the width, height and color channel count of our input images.
It takes images of 224x224 pixels, which means that our images first need to be cropped and resized to fit the model.
Also note that it takes floats and not integers as inputs.
That is because the inputs need to be normalized to a range of [-1.0, 1.0] from the original [0, 255] before the model can handle it.

We also see that it outputs 3 numbers for each frame.
These are the class probabilities, one for each class.

A Tensorflow Keras-model usually consists of multiple layers, but we currently only see one.
This is because the model is actually multiple models wrapped into one graph.
To visualize this go to https://netron.app/ and drag your model file into the page.
It should show you a graph with one or more "sequential_x"-blobs. If you click on those, it will drill into the sub-models and from there you can even look at the single layers and operations that are done on the graph.
Look around if you want.
We find that Teachable Machine actually produces a special kind of neural networks: https://en.wikipedia.org/wiki/Convolutional_neural_network

If you want to get a deeper understanding of what happens inside a neural network, you can find a nice visualization here:
https://playground.tensorflow.org/

### Stage 3: Self-trained Keras model
In the last stage we used the help of an external tool to create our datasets and model.
We will now replicate that on our own so that we can everything without a web browser.

#### Setup/Usage
Start the program again and select Custom ML Tetris.
The program will prompt you if you want to create train and test data.
Click Yes.

A new window should show up which shows your webcam image.
Above it you see the instructions for creating your datasets, much similiar to the ones from Teachable Machine:
If you press and hold the L-key on your keyboard, the current frames will be taken as examples for the "Left"-class.
Think of a movement or hold an item into the camera that should represent the Left-class and hold L.
Vary a bit and release the key when you are satisfied.
You should need at most 400 examples, usually less suffice.

Repeat the same with the R-key for Right and the T-key for Turn.
Also use the A-key for recording "Other"-frames that don't represent a class but might be similiar to the ones above.

Press Escape when you are happy. What now happens in the background is that your files will be divided into three distinct datasets: train, validation and test.

The files in the train-dataset are used as examples for your training. Files from the validation-set will not be seen by the neural net during training, however they are used in the training process for checking the progress and performance of the training. The test-set is used after training has concluded so create performance metrics on completely new data.

In the next step, the program will prompt you to retrain the model.
Answer with "Yes".
You should see in your console the progress on the training.
Model training happens in so called "epochs": In each epoch, each training sample is seen by the neural net about once.
After the end of each epoch, the training program will check the current performance of your model against the validation dataset.
When the model has not improved anymore for some time or when 25 epochs have been reached, it will stop and save the model to a file.

A new window will pop up that shows you how the accuracies and losses and developed over the epochs during training.

Now it's time to play. Have fun!

#### Background
The datasets are saved into `data/train_data`, `test_data` and `validation_data` respectively.
Inside you will find one folder for each of the classes.

The model is saved to `data/models/simple_tf/tf_classification.h5`.
We already know that format from the previous stage - it contains the models architecture (how the neurons connect) and the weights (which we have trained).
You can again look at it using https://netron.app if you want.

![Simple CNN](/docs/typical_small_cnn.png)

Take a look at `src/tf_classfication.py`.
This contains all the basic ingredients of a neural net training:
* dataset management
* model creation
* model training
* evaluation

For the dataset management we use the ImageDataGenerator from Keras.
We provide it with the preprocessing function that gives us the preprocessed images (save the resizing).
The generator also gives us the possibility to change images on fly by slightly altering them (called Augmentation), resize them, reading them in a random order and organizing them into batches.
We do the augmentations only on the training data and don't shuffle the test data randomly.
The generator expects the images to be organized as we have it, with each class in its own folder.
The generators can be used like any Python iterator.

Next, the model is created: We use the Keras Sequential model for that, where the single layers are organized neatly one after another.
On the first layer, we have to provide the program with the expected image formate: images with 3 color channels and a size of 224x224 pixels.
Each layer type has their own parameters:
* For the convolutional layers, we need to provide the number of filters, the kernel size, the padding strategy and the activation function.
* For max pooling layers we need to provide the pool size, strides and padding (we usually leave that at defaults here).
* For the dropout layer, we need to provide the dropout rate.
* For the dense layers, we need to provide the number of neurons and the activation function.
The last layer should have the same number of neurons as we have classes and the softmax-activation function: This will make you neural network outputs always sum up to 1 exactly, so we can handle the outputs like relative confidences.

The training part starts off with the definition of various Callbacks: These are program parts that are called on events during the training and which can then react:
* The GUICallback will just show the training progress in the GUI
* the EarlyStopping-callback checks wether the training has stagnated for some time and if so, stops the trainign early
* The ReduceLROnPlateau-callbacks also checks for stagnant training and adjusts the learning rate to be able to learn smaller differences between classes.
* The ModelCheckpoint-callback saves the model periodically.

After that we need to compile the model with an optimizer, a loss-function and and target metric.
These can usually be left alone here.

The call to `model.fit()` then starts the actual training.

In the end we show how the losses and metrics have evolved over the epochs for the training- and validation set and evaluate the model: You will see a so called [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) that shows you how many samples have been recognized as which class and what would have been the actual class.
In the ideal case, all entries but the diagonal ones should be zero, else some samples have been mis-classified.

## Troubleshooting: My model does not work!
Things to try out if your model does not match your expectations, especially in the last stage:
* Make sure that your classes are well-defined and don't overlap
* Play with the layers: Increase or reduce their sizes or even remove or add some
* Play with the training parameters: Learning rate, batch size, 
* Stand centrally in front of your camera



## Credits

This package was created with Cookiecutter and the [sourcery-ai/python-best-practices-cookiecutter](https://github.com/sourcery-ai/python-best-practices-cookiecutter) project template.
