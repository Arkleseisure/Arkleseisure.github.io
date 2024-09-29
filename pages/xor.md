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
[[-0.5674,  0.2844],  
[-0.8620,  0.8800],  
[ 0.7420, -0.9673],  
[-1.0114,  1.0114]]  
 ***2nd layer***:  
[[-0.6669,  0.7351,  0.8377,  1.0199]]  

 **Biases**:  
 ***Hidden Layer***:  
 [ 5.6738e-01, -1.8065e-02,  5.6761e-01, -7.1264e-08]  
 ***Output***:  
[-0.0971]   

 Now let's dissect how this network works (remember we're using ReLU activations, so negative results are set to 0):  
 Neuron 1 (Weights [-0.5674,  0.2844], Bias 5.6738e-01):  Activates when the second input is 1  
 Neuron 2 (Weights [-0.8620,  0.8800], Bias -1.8065e-02): Activates when the second input is 1, but not when both are.  
 Neuron 3 (Weights [ 0.7420, -0.9673], Bias 5.6761e-01):  Activates slightly when both are 0, more when first input is 1, less when the second input is also 1.  
 Neuron 4 (Weights [-1.0114,  1.0114], Bias -7.1264e-08): Activates when the second input is 1 but not the first.  

 Now the output is constructed from these as follows (Weights [-0.6669,  0.7351,  0.8377,  1.0199], Bias -0.0971):  
 Neuron 1 suppresses the output of (0, 1) and (0, 0)  
 Neuron 2 contributes for the input (0, 1)  
 Neuron 3 contributes for the input (1, 0) and (0, 0)  
 Neuron 4 contributes for the input (0, 1)  

We see that Neurons 2 and 4 are essentially redundant: they each contribute to the output and all we need is a linear combination of them to produce the correct thing. Equally we see that Neuron 1 is there to adjust the output to a reasonable number. In other runs, we often see some neurons dieing (i.e never activating), allowing the network a more condensed interpretation. 

## Interpreting XOR Neural Networks  
What we've just done is what I mean when I say "Fully interpreting" neural networks. We understand every single aspect of how it works in an intuitive manner. This is rather painful to do by hand, as I'm sure you know if you just carefully followed what I did (don't worry I really won't judge you if you just skipped to the end). It would be really nice if we could automate it. Thankfully I think my friend ChatGPT can help. If we can give him the right information and prompt him correctly, we'll see what he can do.

First though if we want this to be a fully scalable method, we need something which will work better than just a set of weights - much as these are technically enough to figure out what the network is doing, for anything more than the 4 neurons we just discussed it's just plain confusing. 

For this we'll consider the concept of activation maximization. This is the process where an input is optimised to maximise the output of one specific neuron in the network. The purpose of this is to understand what inputs trigger this neuron and by extension what the activation responds to. In code, this is surprisingly simple to do: you set the input image to noise, tell the optimiser that this image contains the parameters it needs to optimse, and set $loss = -activation + regularisation$. The regularisation used here is the L2 norm, and is needed to ensure that the input values don't go crazy. Although not that interesting in the context of the XOR problem, especially with only one layer, we'll see with MNIST later that activation maximisation makes it far easier to interpret image classifiers.

Another piece of information which is useful for understanding larger networks is looking at the type of inputs that the neuron activates for. For XOR, this is as simple as just listing the 4 possible inputs and the activations of the neuron for each of them, but for larger networks we shall have to become more creative, as we'll see with MNIST.

## The Prompt
If we now return to the neural network I analysed earlier, we're going to see how one might go about prompting chatgpt to come to the same conclusions.

### Background prompt
First, we need to ensure that ChatGPT knows what sort of network it's dealing with:
*The neural network is an mlp which has 1 hidden layers and width 4. It has final loss 6.646488914796578e-13 (loss function <class 'torch.nn.modules.loss.MSELoss'>) and relu activations.*  

### Neuron wise prompt
When we analysed the xor network earlier, the easiest way to understand it was to look at it neuron by neuron. If we organise the information in a similar way, it will be easier for ChatGPT to interpret it in the same way. We go through the layers and list the neurons for each layer. Then for each neuron we include:  
Weights  
Bias  
Response to inputs  
Maximising input  
This looks something like the following:

This is for layer 1:  
 These are the weights for feature 1:  
 [-0.5673814415931702, 0.2843979299068451]  
 This is the bias for feature 1:  
 0.5673814415931702  
 This is the feature's response to some input data:  
 Input: [0, 0]  
 Feature response: 0.5673814415931702  
 Input: [0, 1]  
 Feature response: 0.0  
 Input: [1, 0]  
 Feature response: 0.567607581615448  
 Input: [1, 1]  
 Feature response: 0.0  
 This is an input designed to maximise the output of the feature:  
 Input: [-0.28368857502937317, 0.1421663910150528], Output activation of feature: 0.7687729001045227  
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.  

We ask chatgpt at the end to explain its interpretation so we can pick it apart. This prompt is then repeated with data for the rest of the neurons.  

  ### Final layer prompt
Understanding how the intermediate neurons work isn't sufficient to understand how they're combined into making the final output, so for this we need a final addition tot the prompt. First we include the weights and biases as before, so that it can understand how the previous layer connects to the output. Then, we add the outputs for each input of the xor function so that it can evaluate where the network works and where it doesn't.
  
 Once you have found the values of these features, use the following data to summarise how you think the rest of the network works:  
 Final layer weights: [[-0.6669246554374695, 0.7351352572441101, 0.8376786112785339, 1.019875168800354]]  
 Final layer biases: [-0.09707102924585342]  
 Outputs:  
Input: [0, 0], Output: 1.0207295417785645e-06  
Input: [0, 1], Output: 0.9999996423721313  
Input: [1, 0], Output: 0.9999988079071045  
Input: [1, 1], Output: 3.725290298461914e-07  
 Finally summarise overall how the network works. 

 We include the summary prompt in the hope that if ChatGPT is good enough at evaluating the network, we'll just be able to read the summary at the end and understand how the network works.

## ChatGPT response
Now finally let's have a look at how ChatGPT responds to our prompt for the neural network we analysed above.

 To interpret the functioning of this neural network, let's break down the roles of each feature in the hidden layer, their contributions, and how the final layer synthesizes these to produce the output.

### Feature Analysis
#### Feature 1
- **Weights:** [-0.567381, 0.284398], **Bias:** 0.567381
- **Behavior:**
  - Produces a fixed positive value (0.567381) when \(x_1 = 0\) and \(x_2 = 0\) or \(x_1 = 1\) and \(x_2 = 0\).
  - Produces a value of 0 when \(x_2 = 1\), due to the ReLU activation negating negative inputs.
  - Input [-0.283688, 0.142166] yields the highest output, suggesting an optimization around reversing weights effects.
- **Interpretation:** Feature 1 is likely modeling an interaction where presence of \(x_2\) negatively impacts the response when weights are combined. It produces positive outcomes primarily from scenarios where \(x_2 = 0\).

#### Feature 2
- **Weights:** [-0.861964, 0.880029], **Bias:** -0.018065
- **Behavior:**
  - Responds with values influenced strongly by \(x_2\), nearly unaffected by \(x_1\) (e.g., 0.861964 when \(x_2 = 1\) regardless of \(x_1\)).
  - Input [-0.430972, 0.439968] yields maximum response.
- **Interpretation:** Feature 2 captures scenarios sensitive to changes in \(x_2\), almost acting as distinguishing presence of \(x_2\). When both inputs are high, feature outputs are also maximized indicating interactions between both variables.

#### Feature 3
- **Weights:** [0.742047, -0.967348], **Bias:** 0.567608
- **Behavior:**
  - Dominantly activates when \(x_1 = 1\) and \(x_2 = 0\), producing high output.
  - Nullifies when \(x_2 = 1\), likely driven by the negative weight on \(x_2\).
  - Maximal input [0.370990, -0.483720].
- **Interpretation:** Feature 3 heavily weights positive \(x_1\) conditions while negating effects from \(x_2\). Sensibly aligns with detecting a unique meaning of conditions isolated to \(x_1\).

#### Feature 4
- **Weights:** [-1.011382, 1.011382], **Bias:** -0.000000071263649204
- **Behavior:**
  - Activates minimally under all test inputs due to almost balancing weights.
  - Best maximizes around balanced input [-0.505692, 0.505648].
- **Interpretation:** Feature 4 represents a delicate balancing scenario, effectively acting when the input vector's characteristic aligns near-zero influence (given tight weight symmetry).

### Final Layer Synthesis
- **Final Weights:** [-0.666925, 0.735135, 0.837679, 1.019875], **Bias:** -0.097071
- **Outputs Analysis:**
  - The outputs are mapped binary-like: near 0 or near 1, effectively encoding state changes similar to XOR-like behavior.
  - Inputs \(x = [0, 1]\) or \(x = [1, 0]\) output ~1, whereas \(x = [0, 0]\) or \(x = [1, 1]\) are near zero.

### Overall Network Summary
This MLP is designed to perform pattern recognition akin to an XOR-like operation on two binary inputs. Each feature is crafted to separately encapsulate discrete relationships between the input bits, mediated through targeted weight and bias structures followed by ReLU activations. The final layer aggregates these feature responses into a cohesive binary output, effectively acting as a logical decision mechanism matching XOR dynamics by contrasting paired input states.
