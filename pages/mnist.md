---
layout: default
title: Home
---

# MNIST Neural Networks  
## The MNIST dataset 
As we noted on the XOR page (if you haven't read that yet, it might be best to do that first), failure on the XOR dataset killed machine learning research off for a while after Minsky and Papert's infamous book was published in 1969. The source of MNIST's notoriety is similar, in that success on the dataset in 1998 showed that machine learning could be used for real world tasks, and kicked the field off properly.

MNIST is a dataset of handwritten digits, all in greyscale. They look something like this:  
![Image of a pixelated 9, in greyscale](/images/mnist_example_image.png "An MNIST image")

The goal of course is to classify the digit correctly. For this, we have 10 outputs, each corresponding to a possible digit. These outputs are then passed through the "softmax" function, which turns them into a probability distribution. Mathematically, this is defined as $softmax(x_i) = \frac{e^{x_i}}{sum(e^{x_j})}$. To keep interpreting the results as simple as possible, we will once again use a simple fully connected network.

## Interpreting an MNIST Neural Network
When we looked at the XOR dataset, we saw that we were able to interpret the network just from its weights. This doesn't work for MNIST. Activation maximization, also introduced with XOR, does however. It also produces beautiful images like this one.  
![I](/images/mnist_trigger_image.png "Activation maximization resulting image")

Based on the analysis of the MLP neural network with one hidden layer and the given feature activations, here's a summary of the most important feature from that layer:

### Feature 1:
- **Captures**: The neuron in this feature captures the presence of smooth, rounded shapes or arcs, generally found in digits with circular or elliptical characteristics.
- **Activation Patterns**:
  - It shows a high mean activation for the digit "0", indicating a strong response to circle-like shapes.
  - Moderate activations are observed for digits "6" and "3", which also contain some rounded parts, though not as complete as "0."
- **Image Features**: The neuron is sensitive to continuous, smooth arcing lines that typically make up circular or elliptical components of digits.
- **Neural Representation**: This representation likely focuses on identifying and responding to any circularity or arc in the input, critical for recognizing digits that inherently have a round shape.
- **Importance**: Since it potentially detects the full or partial presence of circles, it might contribute significantly to the model's ability to differentiate digits like "0" from others such as "1" or "7," which lack these attributes.

Dead neurons (features 0, 2, 3, 4, and 5) don't contribute to the model's output since they don't respond to any input pattern meaningfully. Therefore, Feature 1 is the primary contributor in this layer for recognizing curved or rounded shapes in the MNIST dataset. Further improvement of the network could involve augmenting other forms of shape detections or re-examining the architecture to ensure more neurons are active and contributing insightfully to the task.
Summary: The neural network is a single-hidden-layer MLP trained on the MNIST digit recognition task. The hidden layer consists of 6 neurons, with only "Feature 1" being significantly active, which responds to smooth, rounded shapes primarily seen in digits like "0", and to a lesser extent in "6" and "3". The rest of the neurons (features 0, 2, 3, 4, and 5) do not meaningfully activate for any input patterns, severely limiting the model's capacity to learn diverse digit characteristics.

Final outputs are primarily influenced by Feature 1's activation and its respective weights to each output class. Despite this primary reliance, the variation in weights causes a different ordering of contributions to output activations:
- Outputs 0, 1, 2, and 3 see a mix of both positive and negative influences from the all activations.
- Outputs 4, 6, 7, and 9 are negatively influenced primarily by Feature 1 and positively by differing combinations of deactivated or less effective features like 4 or 0.
- Output 5 balances all activations with moderately positive weights, slightly favoring feature 1.
- Output 8 slightly emphasizes both inactive feature combinations and weaker influences from deactivated neurons, leading to potential misclassifications.

The network's relatively low accuracy reflects its predominant reliance on one active feature and limited learning due to the effective "dead" state of remaining features, limiting discrimination across diverse digit patterns. To improve, activating additional features for varied shape detection is crucial, or revising network architecture is necessary for enhanced representational capacity.


