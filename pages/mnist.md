---
layout: default
title: Home
---

# MNIST Neural Networks  
## The MNIST dataset 
As we noted on the XOR page (if you haven't read that yet, it might be best to do that first), failure on the XOR dataset killed machine learning research off for a while after Minsky and Papert's infamous book was published in 1969. The source of MNIST's notoriety is similar, in that success on the dataset in 1998 showed that machine learning could be used for real world tasks, and kicked the field off properly.

MNIST is a dataset of handwritten digits, all in greyscale. They look something like this:  
![Image of a pixelated 9, in greyscale](/images/mnist_example_image.png "An MNIST image")

The goal of course is to classify the digit correctly. For this, we have 10 outputs, each corresponding to a possible digit. These outputs are then passed through the "softmax" function, which turns them into a probability distribution. Mathematically, this is defined as $softmax(x_i) = \frac{e^{x_i}}{sum(e^{x_j})}$.




