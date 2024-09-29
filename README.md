This is a project for the BlueDot Alignment course attempting to automate interpretability by fully explaining neural networks using ChatGPT.
This is attempted on 2 problems: the XOR problem and the MNIST dataset.

**XOR Neural Networks**  
***The XOR problem***  
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

***An XOR Neural Network***  
To start, we look at a simple xor neural network: 1 hidden layer of 4 neurons, with a ReLU activation function.  
This is sufficient to solve the xor problem (there's 1 hidden layer rather than 1 set of weights linking input and output, which is what Minsky and Papert looked at).  
This network has 12 weights (4 linking the 1st input to the hidden layer, 4 linking the second input to the hidden layer and 4 linking the hidden layer to the output).   
It also has 5 biases (4 for the hidden layer and 1 for the output).  

An example I trained earlier has the following parameters:
Weights:  
1st layer:  
[[-0.5674,  0.2844],  
[-0.8620,  0.8800],  
[ 0.7420, -0.9673],  
[-1.0114,  1.0114]]  
 2nd layer:  
[[-0.6669,  0.7351,  0.8377,  1.0199]]  

 Biases:  
 Hidden Layer:  
 [ 5.6738e-01, -1.8065e-02,  5.6761e-01, -7.1264e-08]  
 Output:  
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

We see that Neurons 2 and 4 are essentially redundant: they each contribute to the output and all we need is a linear combination of them to produce the correct thing. 
Equally we see that Neuron 1 is there to adjust the output to a reasonable number. In other runs, we often see some neurons dieing (i.e never activating), allowing the network a more condensed interpretation. 

***Interpreting XOR Neural Networks***  
What we've just done is what I mean when I say "Fully interpreting" neural networks. We understand every single aspect of how it works in an intuitive manner.  
This is rather painful to do by hand, as I'm sure you know if you just carefully followed what I did (don't worry I really won't judge you if you just skipped to the end).  


