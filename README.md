This is a project for the BlueDot Alignment course attempting to automate interpretability by fully explaining neural networks using ChatGPT.
This is attempted on 2 problems: the XOR problem and the MNIST dataset.

**XOR Neural Networks**  
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

To start, we look at a simple xor neural network: 1 hidden layer of 4 neurons, with a ReLU activation function.  
This is sufficient to solve the xor problem (there's 1 hidden layer rather than 1 set of weights linking input and output, which is what Minsky and Papert had).  
This network has 12 weights (4 linking the 1st input to the hidden layer, 4 linking the second input to the hidden layer and 4 linking the hidden layer to the output).   
It also has 5 biases (4 for the hidden layer and 1 for the output).  

An example I trained earlier has the following weights:  
1st layer:  
[[ 1.0579, -1.0332],  
[-0.0502,  0.2102],  
[-1.0009,  1.0331],  
[-0.5001, -0.4604]]   

 2nd layer:  
[[ 0.9285, -0.3533,  1.0398,  0.3237]]  

 Biases:  
 Hidden Layer:  
 [-6.1993e-08,  9.1451e-01, -1.0491e-08, -3.6663e-01]   
 Output:  
[0.3231]   

 Now let's have a look at 

The neural network is an mlp which has 2 hidden layers and width 4. It has final loss 1.1590728377086634e-13 (loss function <class 'torch.nn.modules.loss.MSELoss'>) and relu activations.
This is for layer {i}:
 These are the weights for feature 1:
 [0.7923692464828491, 0.7923692464828491]
 This is the bias for feature 1:
 -0.792369544506073
 This is the feature's response to some input data:
 Input: [0, 0]
 Feature response: 0.0
 Input: [0, 1]
 Feature response: 0.1299123913049698
 Input: [1, 0]
 Feature response: 0.0
 Input: [1, 1]
 Feature response: 0.0
 This is an input designed to maximise the output of the feature:
 Input: [1.8095879568136297e-06, -3.1153883810475236e-06], Output activation of feature: 0.0
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.
 These are the weights for feature 2:
 [1.2335460186004639, 1.233545184135437]
 This is the bias for feature 2:
 0.1299123913049698
 This is the feature's response to some input data:
 Input: [0, 0]
 Feature response: 0.0
 Input: [0, 1]
 Feature response: 1.3634575605392456
 Input: [1, 0]
 Feature response: 0.0
 Input: [1, 1]
 Feature response: 0.0
 This is an input designed to maximise the output of the feature:
 Input: [0.6167523264884949, 0.616769015789032], Output activation of feature: 1.6515171527862549
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.
 These are the weights for feature 3:
 [0.5874751210212708, -0.23565196990966797]
 This is the bias for feature 3:
 -0.6784002780914307
 This is the feature's response to some input data:
 Input: [0, 0]
 Feature response: 0.0
 Input: [0, 1]
 Feature response: 1.3634583950042725
 Input: [1, 0]
 Feature response: 0.0
 Input: [1, 1]
 Feature response: 0.0
 This is an input designed to maximise the output of the feature:
 Input: [0.8293144702911377, 0.08754566311836243], Output activation of feature: 0.0
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.
 These are the weights for feature 4:
 [-0.0039754509925842285, -0.06616699695587158]
 This is the bias for feature 4:
 -0.5370442271232605
 This is the feature's response to some input data:
 Input: [0, 0]
 Feature response: 0.7923689484596252
 Input: [0, 1]
 Feature response: 2.59700345993042
 Input: [1, 0]
 Feature response: 0.0
 Input: [1, 1]
 Feature response: 0.0
 This is an input designed to maximise the output of the feature:
 Input: [-0.7484385371208191, 1.2751007080078125], Output activation of feature: 0.0
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.
This is for layer {i}:
 These are the weights for feature 1:
 [-1.0765243768692017, 0.6600565314292908, 0.04654008150100708, 0.49690955877304077]
 This is the bias for feature 1:
 0.24961571395397186
 This is the feature's response to some input data:
 Input: [0, 0]
 Feature response: 0.3353652358055115
 Input: [0, 1]
 Feature response: 0.0
 Input: [1, 0]
 Feature response: 0.07427068799734116
 Input: [1, 1]
 Feature response: 0.27423912286758423
 This is an input designed to maximise the output of the feature:
 Input: [0.4071040153503418, 0.4070901870727539], Output activation of feature: 0.9982901215553284
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.
 These are the weights for feature 2:
 [1.441149115562439, 0.34586769342422485, 0.3930438756942749, -0.21143001317977905]
 This is the bias for feature 2:
 -0.4715760350227356
 This is the feature's response to some input data:
 Input: [0, 0]
 Feature response: 1.1495747566223145
 Input: [0, 1]
 Feature response: 0.0
 Input: [1, 0]
 Feature response: 0.8103405237197876
 Input: [1, 1]
 Feature response: 1.2672706842422485
 This is an input designed to maximise the output of the feature:
 Input: [0.7842757701873779, 0.7842877507209778], Output activation of feature: 0.8918289542198181
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.
 These are the weights for feature 3:
 [-0.8664223551750183, 0.5967109203338623, -0.2611600160598755, -0.4636949896812439]
 This is the bias for feature 3:
 -0.0032494489569216967
 This is the feature's response to some input data:
 Input: [0, 0]
 Feature response: 1.1495753526687622
 Input: [0, 1]
 Feature response: 1.7881393432617188e-07
 Input: [1, 0]
 Feature response: 0.8103410601615906
 Input: [1, 1]
 Feature response: 1.2672713994979858
 This is an input designed to maximise the output of the feature:
 Input: [0.3680267632007599, 0.3680248260498047], Output activation of feature: 0.6160563230514526
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.
 These are the weights for feature 4:
 [-0.2515183389186859, 0.8050224781036377, 0.06480950117111206, -0.34149622917175293]
 This is the bias for feature 4:
 0.16965673863887787
 This is the feature's response to some input data:
 Input: [0, 0]
 Feature response: 1.1107803583145142
 Input: [0, 1]
 Feature response: 1.5685653686523438
 Input: [1, 0]
 Feature response: 0.8598846793174744
 Input: [1, 1]
 Feature response: 2.0610077381134033
 This is an input designed to maximise the output of the feature:
 Input: [0.4965061843395233, 0.4965225160121918], Output activation of feature: 1.2603484392166138
 Please give an interpretation of what you think this feature means, and use reasoning based on the data given.
 Once you have found the values of these features, use the following data to summarise how you think the rest of the network works:
 Final layer weights: [[0.5877380967140198, -0.7923039197921753, 0.2806641757488251, 0.31707826256752014]]
 Final layer biases: [-0.3049068748950958]
 Outputs:
Input: [0, 0], Output: 4.470348358154297e-07
Input: [0, 1], Output: 0.9999995231628418
Input: [1, 0], Output: 1.0000001192092896
Input: [1, 1], Output: 1.4901161193847656e-07
 Finally summarise overall how the network works.
