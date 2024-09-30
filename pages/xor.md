---
layout: default
title: XOR
---

# XOR Neural Networks  
## The XOR problem  
The XOR (exclusive or) problem is defined as follows:  
2 binary inputs   
1 output  

0 xor 0 = 0  
0 xor 1 = 1  
1 xor 0 = 1  
1 xor 1 = 0  

Although it seems obvious in today's neural network based society that neural networks can solve this problem, in 1969 Minsky and Papert proved that a single layered neural network could not solve this problem. This was taken by the community as a reflection of neural networks as a whole and so the first "ai winter" ensued. 

Naturally this is no longer the case, and the point of this project is to interpret the full functionality of these xor networks end-to-end.  
Once we understand how to do that, we'll move onto understanding how to get this working on the famous MNIST dataset.  

## An XOR Neural Network  
To start, we look at a simple xor neural network: 1 hidden layer of 4 neurons, with a ReLU activation function.  
This is sufficient to solve the xor problem (there's 1 hidden layer rather than 1 set of weights linking input and output, which is what Minsky and Papert looked at).  
This network has 12 weights (4 linking the 1st input to the hidden layer, 4 linking the second input to the hidden layer and 4 linking the hidden layer to the output).   
It also has 5 biases (4 for the hidden layer and 1 for the output).  

An example I trained earlier has the following parameters:  
**Weights**:  
***1st layer***:  
[[-0.7834, -0.7794],
[-0.9649,  0.7730],
[ 0.4980, -1.2739],
[ 0.1256, -0.4764]]  
 ***2nd layer***:  
[[-1.0414,  1.0336,  0.7825,  0.3891]]  

 **Biases**:  
 ***Hidden Layer***:  
 [ 0.7759,  0.1917,  0.7758, -0.1807]  
 ***Output***:  
[0.0029]   

 Now let's dissect how this network works (remember we're using ReLU activations, so negative results are set to 0):  
 Neuron 1 (Weights [-0.7834, -0.7794], Bias 0.7759):  Activates for input [0, 0]  
 Neuron 2 (Weights [-0.9649,  0.7730], Bias 0.1917):  Activates when the second input is 1, but not when both are.  
 Neuron 3 (Weights [ 0.4980, -1.2739], Bias 0.7758):  Activates slightly when both are 0, more when first input is 1, but not when the second input is also 1.  
 Neuron 4 (Weights [ 0.1256, -0.4764], Bias -0.1807): Dead neuron.  

 Now the output is constructed from these as follows (Weights [-1.0414,  1.0336,  0.7825,  0.3891], Bias 0.0029):  
 Neuron 1 suppresses the output of (0, 0)  
 Neuron 2 contributes for the input (0, 1)  
 Neuron 3 contributes for the input (1, 0) and (0, 0)  
 Neuron 4 doesn't contribute 

We see that Neurons 1 and 3 are essentially redundant: Neuron 1 is there to account for the fact that Neuron 3 imperfectly represents the (1, 0) activation. Otherwise, 

## Interpreting XOR Neural Networks  
What we've just done is what I mean when I say "Fully interpreting" neural networks. We understand every single aspect of how it works in an intuitive manner. This is rather painful to do by hand, as I'm sure you know if you just carefully followed what I did (don't worry I really won't judge you if you just skipped to the end). It would be really nice if we could automate it. Thankfully I think my friend ChatGPT can help. If we can give him the right information and prompt him correctly, we'll see what he can do.

First though if we want this to be a fully scalable method, we need something which will work better than just a set of weights - much as these are technically enough to figure out what the network is doing, for anything more than the 4 neurons we just discussed it's just plain confusing. 

For this we'll consider the concept of activation maximization. This is the process where an input is optimised to maximise the output of one specific neuron in the network. The purpose of this is to understand what inputs trigger this neuron and by extension what the activation responds to. In code, this is surprisingly simple to do: you set the input image to noise, tell the optimiser that this image contains the parameters it needs to optimse, and set $loss = -activation + regularisation$. The regularisation used here is the L2 norm, and is needed to ensure that the input values don't go crazy. Although not that interesting in the context of the XOR problem, especially with only one layer, we'll see with MNIST later that activation maximisation makes it far easier to interpret image classifiers.

Another piece of information which is useful for understanding larger networks is looking at the type of inputs that the neuron activates for. For XOR, this is as simple as just listing the 4 possible inputs and the activations of the neuron for each of them, but for larger networks we shall have to become more creative, as we'll see with MNIST.

## The Prompt
If we now return to the neural network I analysed earlier, we're going to see how one might go about prompting chatgpt to come to the same conclusions.

### Background prompt
*First, we need to ensure that ChatGPT knows what sort of network it's dealing with.*  
The neural network is an mlp which has 1 hidden layer and width 4. It has final loss 2.1766519042124705e-06 (loss function <class 'torch.nn.modules.loss.MSELoss'>) and relu activations.  

### Activation wise prompt
When we analysed the xor network earlier, the easiest way to understand it was to look at it neuron by neuron. If we organise the information in a similar way, it will be easier for ChatGPT to interpret it in the same way. We go through the layers and list the neurons for each layer. Then for each neuron we include:  
Weights  
Bias  
Response to inputs  
Maximising input  

This looks something like the following:

This is for layer 1:  
 These are the weights for feature 1:  
 [-0.7834176421165466, -0.7794498801231384]  
 This is the bias for feature 1:  
 0.7759184241294861  
 This is the feature's response to some input data:  
 Input: [0, 0]  
 Feature response: 0.7759184241294861  
 Input: [0, 1]  
 Feature response: 0.0  
 Input: [1, 0]  
 Feature response: 0.0  
 Input: [1, 1]  
 Feature response: 0.0  
 This is an input designed to maximise the output of the feature:  
 Input: [-0.3917100131511688, -0.38972726464271545], Output activation of feature: 1.386563777923584  
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.  
 These are the weights for feature 2:  
 ...  

We ask chatgpt at the end to explain its interpretation so we can pick it apart. 

### Final layer prompt
*Understanding how the intermediate neurons work isn't sufficient to understand how they're combined into making the final output, so for this we need a final addition tot the prompt. First we include the weights and biases as before, so that it can understand how the previous layer connects to the output. Then, we add the outputs for each input of the xor function so that it can evaluate where the network works and where it doesn't.*
 
Once you have found the values of these features, use the following data to summarise how you think the rest of the network works:  
 Final layer weights: [[-1.0413832664489746, 1.0335971117019653, 0.7824711203575134, 0.38914060592651367]]  
 Final layer biases: [0.002885989611968398]  
 Outputs:  
Input: [0, 0], Output: -3.355019725859165e-05  
Input: [0, 1], Output: 0.9999251365661621  
Input: [1, 0], Output: 0.9995940923690796  
Input: [1, 1], Output: 0.002885989611968398  
 Finally summarise overall how the network works.  

 We include the summary prompt in the hope that if ChatGPT is good enough at evaluating the network, we'll just be able to read the summary at the end and understand how the network works.

## ChatGPT response
Now finally let's have a look at how ChatGPT responds to our prompt for the neural network we analysed above.

To interpret the features and summarize how the network works, I'll examine each feature in the hidden layer based on the given weights, biases, and example inputs.

### Feature 1:  
- **Weights:** [-0.7834, -0.7794]  
- **Bias:** 0.7759  
- **Response Analysis:**  
  - When both inputs are 0, the feature outputs 0.7759 (just the bias).  
  - If either or both inputs are 1, the output is 0. This suggests that the weighted sum is negative and the ReLU activation forces it to 0.  
  - The input [-0.3917, -0.3897], maximizes the output to 1.3866.  

- **Interpretation:** Feature 1 is likely detecting cases where both inputs are close to 0, as it outputs its bias value whenever the input vector is far from positive values in both dimensions. The negative weights suggest that it suppresses any positive input.

### Feature 2:
- **Weights:** [-0.9649, 0.7729]  
- **Bias:** 0.1917  
- **Response Analysis:**  
  - Output is 0.965 when the second input is 1 and the first is 0.  
  - Output is 0 if the first input is 1, reinforcing that the first input's negative weight cancels the potential positive impact from the second input.  
  - It’s maximized with input [-0.4824, 0.3865], further validating its interest in the disparity between both indices with a slight positive from bias.  

- **Interpretation:** This feature likely measures the balance between the two inputs, focusing on when the second input is stronger than an offset defined primarily by larger magnitude of the first. Essentially, it’s activated strongly when the second input dominates the influence.  

### Feature 3:  
- **Weights:** [0.4980, -1.2739]  
- **Bias:** 0.7758  
- **Response Analysis:**  
  - Output is 1.2738 when the first input is 1.  
  - Output is 0 if the second input is 1, as its contribution becomes negative and surpasses the bias.  
  - Maximized with a careful balance, like [0.2490, -0.6369].  

- **Interpretation:** Feature 3 seems to detect cases heavily influenced by the first input being positive while the second component isn’t significantly larger to turn the weighted sum negative.  

### Feature 4:  
- **Weights:** [0.1256, -0.4764]  
- **Bias:** -0.1807  
- **Response Analysis:**  
  - Output is always 0 for the given inputs.  
  - Maximized input still results in 0, indicating it remains inactive given the input ranges observed.  

- **Interpretation:** Feature 4 does not activate with the provided inputs strongly and is possibly redundant or sensitive to values outside the testing domain. It appears to have a low contribution given the weights and biases.  

### Summary of Network Functionality:  
- **Final Layer Analysis:**  
  - **Weights:** [-1.0414, 1.0336, 0.7825, 0.3891], Bias: 0.002886  
  - Outputs showcase a toggle pattern primarily activated when specific features in the first layer respond positively.  

- **Working Hypothesis:**  
  - The network likely operates to distinguish binary patterns where it is paramount to have a presence of activation from particular features.  
  - The outputs approximate 1 when the first or second input is dominant (but not both), reinforced by the positioning of responses around the units dimension of either feature 2 or 3.  
  - The bias is non-effectively low to maintain neutral as additional output logic, assisting primarily as a balancing tool.  

In conclusion, this MLP network is structured to predominantly activate and make distinctions amongst contrasting input patterns which represent unique configurations, relying heavily on alignment of inputs well enough to ‘trigger’ specific ordinance in its narrow-weighted hidden features, leading to the targeted outputs.  

## Conclusion  
Clearly we see that ChatGPT is well able to interpret the activations of the hidden layer neurons, but seems to have difficulty being precise enough in providing a summary. I believe it is likely that this is solvable with better prompts, and I leave that to further research, although some additional techniques have been used with the MNIST dataset.  
