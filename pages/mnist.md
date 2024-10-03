---
layout: default
title: MNIST - Sean Herrington BlueDot project
---

# MNIST Neural Networks  
## The MNIST dataset 
As we noted on the XOR page (if you haven't read that yet, it might be best to do that first), failure on the XOR dataset killed machine learning research off for a while after Minsky and Papert's infamous book was published in 1969. The source of MNIST's notoriety is similar, in that success on the dataset in 1998 showed that machine learning could be used for real world tasks, and kicked the field off properly.

MNIST is a dataset of handwritten digits, all in greyscale. They look something like this:  
![Image of a pixelated 9, in greyscale](/images/mnist_example_image.png "An MNIST image")

The goal of course is to classify the digit correctly. For this, we have 10 outputs, each corresponding to a possible digit. These outputs are then passed through the "softmax" function, which turns them into a probability distribution. Mathematically, this is defined as $softmax(x_i) = \frac{e^{x_i}}{sum(e^{x_j})}$. To keep interpreting the results as simple as possible, we will once again use a simple fully connected network.

## Interpreting an MNIST Neural Network
When we looked at the XOR dataset, we saw that we were able to interpret the network just from its weights. This doesn't work for MNIST. Activation maximization, also introduced with XOR, does however. It also produces modern art, such as this image:  
![I](/images/mnist_trigger_image.png "Activation maximization resulting image")

What does this image represent? Well, to my eye, this looks like this neuron activates for the loop that appears at the bottom of some digits, such as in the digits 3, 5, 6, and 8. In particular, we can see that the dark section towards the middle hinders any digits which might otherwise activate the surroundings of this bottom loop, such as the digit 4 for example. Equally noteworthy is the bright spot towards the top right of the image, which may be an example of polysemanticity in action - this example was taken from a very small network with a single hidden layer of 6 neurons. I imagine that this particular spot identifies images which reach far out into that corner, potentially helping to identify digits such as the number 7, which may activate that part of the image.

Now, a note on polysemanticity. This is a phenomenon which has been noticed in machine learning interpretability research where neurons activate for more than one type of feature. It complicates interpretability research considerably and isn't really addressed in this project, although I expect that it would be in principle possible to introduce this into our methods. This could be done for instance by producing several activation maximising images, but instead of maximising a single activation, we attempt to maximise the magnitude of the activation vector, and minimise the scalar product between these activation vectors, hence finding inputs which correspond to very different semantic meaning inside the network. Including prompts talking about the literature around polysemanticity may also help. The details of implementation of this are left to future work.

Just as some neurons have multiple different meanings, some have none. These we call 'dead neurons', and are 0 for any given (normal) input. The reason for their existence is not that this is the optimal way to solve the problem, but rather a random result of the training process, where in ReLU neural networks some neurons get trapped in this state (I am currently doing a separate project researching this pheneomenon). For our purposes, any neuron which doesn't activate at all during testing will be considered dead and automatically labelled as such, without wasting valuable chatgpt time. 

## Similarities and differences with XOR
As a reminder, for the XOR neural network we gave ChatGPT the following information:  
1. Background prompt (info about the general network)   
2. Activation wise prompt (prompt with information about each of the activations)  
3. Final layer prompt (explain how the final set of activations connects to the output)

For MNIST, the following changes are made:
1. Each activation prompt is sent to the API separately. This is due to size restrictions which can start occurring, especially since language models are known to work less well when large prompts are present.
2. Each activation prompt contains the background prompt for the network.
3. Layer wise prompts are now used to condense the activation responses into a single useable list which contains the functionality of each activation in the layer.
4. Activation prompts from later layers only include the top 10 most important connections to the previous layer (in terms of weight magnitude), so as to condense the size of the prompt.  
5. Activation prompts include a prompt describing the average output of each of the output classes.
6. The final layer/summary prompt now takes in the condensed layer responses and the most important weights for the final layer (again, the top 10).


## Background prompt
An example background prompt is the follows:

The neural network we are evaluating has the following properties:  
It is an MLP neural network with 1 hidden layers and width 6.  
It is trained on the MNIST dataset and uses relu activations.  
The train accuracy of the final model is 0.61175, the train loss is 1.1200610458977949. The test accuracy of the final model is 0.6305, the test loss is 1.0969773648650782.

The key thing is that all the main points that might not otherwise be known about the network are addressed.

## Activation prompt
A prompt for a single activation would consist of the background prompt, followed by

**Activation specific background:**  
*This just gives context as to where in the network the particular activation it is evaluating is.*  
Your task is to evaluate what the following activation does.  
This activation is at depth 0, with the first layer outputs being labelled as depth 0.  

**Activation mean and standard deviation for each output class:**  
*These give intuitive information about what sort of features the activation might be picking up on from which sorts of numbers it usually contributes to.*  
The following are the average and standard deviations of the activations for each output class:  
Avg label 0: 16.760561412142724, std: 5.126875526036485  
Avg label 1: 0.6375665275972019, std: 1.054784035971524  
Avg label 2: 1.4199343871325254, std: 2.9929772723077956  
Avg label 3: 1.9105483966589976, std: 3.2013420620190147  
Avg label 4: 19.85618057470212, std: 5.544845653758414  
Avg label 5: 9.586604956083182, std: 5.6778547974574884  
Avg label 6: 10.000887012240863, std: 3.1338584110363263  
Avg label 7: 14.530015394239143, std: 5.218367585080105  
Avg label 8: 8.26626239335608, std: 5.2008824643134455  
Avg label 9: 18.319867913045137, std: 5.739438847200249  

**Activation maximisation prompt:**
*As described above, this is very useful.*  
The following image is an input specifically designed to maximise the output of the neuron. *(The image would then be one such as shown above)*  

**Task prompt**
*Giving a precise explanation of a task with examples allows the network to give a more tailored response.*  
Please give an interpretation of what you think this feature means, and use reasoning based on the data and the image given.
The interpretation must be precise enough to be able to accurately reconstruct which images would trigger this neuron, e.g., 'This neuron activates for the 90 degrees of curve from 0 to 90 degrees.'

## Activation response
This is an example response to the activation prompt:

Based on the given activations and image, the neuron appears to highly activate for digits resembling a "4" or "9." Here's the reasoning and interpretation:

1. **High Activation Values for Labels 0, 4, and 9**:
   - The neuron shows high average activations for labels 0, 4, and 9, with particularly high values for label 4 (19.856) and label 9 (18.319).
   - This suggests that the feature captured by this neuron is strongly associated with some part of these digits.

2. **Image Analysis**:
   - The visual pattern observed in the image seems to resemble curved shapes with a vertical component typical of digits "4" and "9."
   - The circle and vertical line could correspond to the typical shapes found in these digits, such as the top loop of the "9" and the vertical and horizontal lines of the "4."

Based on these observations, the neuron likely activates for features resembling the combination of vertical and curved lines typical of certain parts of the digits "4" and "9." Images containing such patterns will likely trigger this neuron. This activation pattern suggests sensitivity to specific lines and shapes common in these numbers.

The neuron we are examining appears to be highly activated by specific patterns found in images corresponding to certain digits. Based on the data provided:

- The highest average activation occurs for digits 7 (20.76), 9 (19.03), and 4 (14.13).
- The lowest activations are for digits 6 (0.23) and 2 (1.02).

From the visualization of the maximum activation input image, it seems to capture features aligned with particular structural nuances common in certain digits. Here's the interpretation of the feature likely emphasized by this neuron:

### Interpretation:
The neuron likely activates for images containing a vertical line with a horizontal line crossing it, similar to a "T" shape, which is a common characteristic in digits like 7 and parts of 4. It might also capture more complex shapes with this feature, affecting how it responds to similar crossings in 9, possibly at an angle or curvature.

### Reconstruction Aids:
1. **Digit 7**: The neuron can be triggered by the descending vertical stroke completed with a left-facing horizontal line on the top, resembling the structure of 7.
2. **Digit 4**: Involves intersecting horizontal and vertical strokes often seen in the top half of the numeral 4.
3. **Digit 9**: May be responding to the loops or curves starting from the vertical portion of 9.

The feature seems less representative of curves without accompanying vertical strokes (hence lower activation for 6 and 2). Designing images with strong vertical-horizontal intersections will likely activate this neuron, especially focusing on the top-left part of the digit canvas possibly, as suggested by the provided visualization image.

## Layer prompt

**Background prompt**
This is the same as before

**List of Features and explanations**
*Here we feed the summaries of the feature activations that have been found in the layer, e.g:*  

The following is the summary of each activation that you gave:
Feature 0:
Dead neuron

Feature 1:
Dead neuron

Feature 2:
*Summary given by ChatGPT of feature 2*

Feature 3:
Dead neuron

Feature 4:
*Summary given by ChatGPT of feature 4*

Feature 5:
Dead neuron

**Task Prompt**  
*We then give a task prompt, giving specific information about how to do this summary.*  
Now please give a summary of the most important features in the layer, feature by feature, and include all the details critical to predicting its activation. Feel free to ignore unimportant neurons such as dead neurons, and please include connections to previous layers in later layers.  
Example:  
Feature 1: Captures the horizontal top of digits, activating for digits such as 5 and 7 but not for the digit 4.  
Feature 2: Captures the circular nature of digits such as 0, made by positive weights from features 4 (left curve on right side of image) and 6 (right curve on left side of image) from the previous layer. It also has negative interference from features 3 and 7 (left curve on left side of image and right curve on right side of the image).   
Feature 4: Captures the absence of a center, made by a negative weight to feature 1 from layer 0 (presence of a center), which is amplified by other features such as a negative weight to feature 3, which captures the presence of a solid line down the middle.  

## Layer response
Here is an example response from ChatGPT.

Layer 0 summary:
Based on the properties and analysis provided for the MLP neural network's activations in the hidden layer, here is a summary of the most important features, with details critical to predicting their activation:

### Feature 2:
- **Activation Characteristics**: This neuron appears to activate strongly for patterns commonly found in digits "4" and "9."
- **Pattern Details**: The high activation suggests sensitivity to vertical and curved line combinations. Specifically, it is likely responding to:
  - The vertical line often found in both digits.
  - The curved portion that could represent the top loop of "9" or parts of "4."
- **Digit Activation**: High average activation values for labels 4 and 9, supporting the interpretation of sensitivity to these features.

### Feature 4:
- **Activation Characteristics**: This neuron is highly activated by structural patterns found in digits such as "7," "9," and "4."
- **Pattern Details**: The neuron captures features resembling a vertical line intersecting with a horizontal line, akin to a "T" shape. This is typical for:
  - The top stroke in digit "7" and "4."
  - Curved shapes in parts of "9."
- **Digit Activation**: Highest average activation for 7, 9, and 4 indicates a strong response to these geometric structures.
- **Inactivation Patterns**: Lesser activation for more rounded shapes without vertical elements, such as "6" and "2."

### Summary:
The two active neurons in this layer contribute significantly to the network's ability to distinguish digits based on specific geometric features:

- **Feature 2**: Primarily responds to combinations of vertical and curved lines, making it critical for recognizing "4" and "9".
- **Feature 4**: Sensitive to intersecting vertical and horizontal lines, key for understanding the structure of "7," "4," and "9."

These neurons effectively capture essential structural features within the digits, aiding the model's decision-making process despite the relatively low accuracy and presence of dead neurons. Improving the network might involve refining these features or expanding capacity with additional functional neurons to enhance pattern recognition and prediction accuracy.

## Summary prompt
Our final prompt is a summary prompt, which should prompt ChatGPT to provide us with a summary sufficient to understand how the neural network works, by combining the information from the layer summaries with their connections to the outputs of the network, to give a full impression of how the network actually finds the answer.

**Background prompt**  
*As always, we start with the background prompt.*  

**Layer summaries**  
*We then add the summaries of each of the layers to the prompt.*

You have summarized the functionality of each layer of the neural network.  
Here they are:  
Layer 0:  
*Layer 0 summary*  
Layer 1:  
*Layer 1 summary*  

**Final weight summaries**  
*At the end we include connections to the outputs.*  

The most important connections for the final layer are:  
For output 0:  
Activation 4, weight: -0.6960952877998352  
Activation 2, weight: 0.6317987442016602  
Activation 0, weight: -0.39977529644966125  
Activation 1, weight: -0.2734476923942566  
Activation 5, weight: 0.26809853315353394  
Activation 3, weight: -0.11677006632089615  
For output 1:  
Activation 2, weight: -1.276571273803711  
Activation 4, weight: 0.8016878962516785  
Activation 1, weight: -0.3857203423976898  
Activation 5, weight: 0.10839473456144333  
Activation 0, weight: -0.03390675038099289  
Activation 3, weight: -0.029084959998726845  
For output 2:  
Activation 1, weight: -0.3801165223121643  
Activation 3, weight: 0.3356472849845886  
Activation 2, weight: -0.20018230378627777  
Activation 5, weight: 0.1808502972126007  
Activation 0, weight: 0.05247001722455025  
Activation 4, weight: -0.050515320152044296  
For output 3:  
...  

**Task Prompt**  
*Of course, we finish off with a task prompt explaining with examples how to summarise the network.*  
Concisely and precisely summarize how the network as a whole works, including specific influences of previous layers on later ones. Make sure to specify how each possible final output is decided upon from the penultimate layer.  
Example for an XOR network with 1 layer of 4 neurons:  
Summary: Feature 1 and 2 activate when the first and second inputs are active respectively, while Feature 0 doesn't contribute and Feature 3 increases with the total number of 1s. The final output is formed by activating the output if either feature 1 or 2 is active, and suppressing it if feature 3 is active.  

## Summary response
*Example response from ChatGPT.*  
### Summary of Network Functionality:  

**Layer 0: Hidden Layer Analysis**  
The hidden layer with 6 neurons uses Rectified Linear Unit (ReLU) activations to identify and capture specific geometric features from the MNIST digit dataset.

- **Feature 2**: This neuron strongly activates for configurations resembling the vertical and curved lines, notably seen in digits "4" and "9." Its activation influences decisions primarily regarding these digits.
- **Feature 4**: Reacts to "T" shaped intersections typical in "7," "9," and "4," playing a crucial role in decoding these patterns.
- Other neurons may have lesser or varied contributions based on unmentioned features. The network's low train and test accuracy likely reflect ineffective neuron activations or dead neurons that do not contribute.

**Final Layer: Decision Making Weighted Analysis**

Each output of the neural network is determined by weighted contributions from the previously mentioned features:

- **Output 0**: Relies mostly on Feature 2 with positive weight, and Feature 4 with negative weight, suggesting it likely distinguishes digits based on lack of "T" effect.
- **Output 1**: Predominantly influenced by negative weight from Feature 2 and positive from Feature 4, targeting digits that prominently exhibit "T" shapes (like "4," "7").
- **Output 2**: Lightly balanced with negative correlations to most features, implying a broader scope for detecting patterns beyond decisive characteristics.
- **Output 3**: Mild influence with positive preference for Feature 4 possibly for digits featuring subtler intersections.
- **Output 4**: Strongly relies on Feature 5's positive weight, implying attention to less-defined patterns not overly influenced by "T" structures.
- **Output 5**: Combines inputs, notably featuring positive weight from Feature 2, to decide among potentially curvilinear related digits.
- **Output 6**: Reactive rather heavily on negative weight from Feature 4, therefore likely avoiding digits like "7" or "4".
- **Output 7**: Emphasizes both positive contributions from Features 4 and 5, leaning towards possibly recognizing digits that combine intersections without strongly vertical features.
- **Output 8**: Appears disparate with reliance on negative weights from most features, gathering insight into neutral pattern territories.
- **Output 9**: Moderate focus on Feature 4's positive weight and other light contributions, employing weighted averages to detect uniqueness in "9".

Overall, while the summation of weighted activations from the hidden layer informs the final decision, the described set of connections reflects an under-capacity network struggling to consistently extract discriminating feature representations across the dataset, contributing to its modest accuracy.

## Summary
So clearly ChatGPT picks up on something, but the process of analysing an MNIST neural network mechanistically is much more painful, and so we haven't done it here. However, we have come up with a testing process to evaluate how much information about the operation of the neural network can be retrieved from ChatGPT's condensed summaries. This is explained in the Testing section.
