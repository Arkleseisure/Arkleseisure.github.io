---
layout: default
title: Home
---

# Testing
## How to test interpretability
If you've followed my writing as intended so far, you will have seen how we were able to prompt GPT4o to interpret the meaning of neurons for neural networks solving both XOR and MNIST datasets. We now need to test how well these interpretations work. Here, we take that to mean "If you are given an input, do you understand the working of the network well enough to predict what its output will be?". For the XOR dataset, this is not really possible, since there are only 4 possible inputs and the answers to these are already known. For MNIST however, we can fairly simply test it. As the API has no recollection of past conversations, it can be given the summary of the neural network (we also include the layer summaries), and an image, and it can then be asked what the network would predict for that image.

## Countering bias
The issue with a naive implementation of this strategy, such as randomly taking a set of images and comparing answers, is that the network will mostly be right, and so ChatGPT can get a high score just by guessing the correct answer each time. For this reason, we test it on a dataset purpose made after network training with 50% correct predictions and 50% incorrect predictions. Statistics are then recorded for each dataset, resulting in 5 fractions:  
**Correct dataset:**  
Correct fraction (C)   
Incorrect fraction (I)  
**Incorrect dataset:**  
Correct about network prediction fraction (N)  
Correct about actual label fraction (L)  
Incorrect on both counts fraction (F)  

We next calculate 2 interpretability scores.  
The first is the correct score: $\frac{C - L}{1 - L}$  
The second is the incorrect score: $\frac{N - \frac{F}{8}}{1-\frac{F}{8}}$  

The correct score evaluates how much more the correct label is predicted when the network prediction is correct, and the incorrect score evaluates how much more the incorrect label that the network predicted is predicted than any other random incorrect label.

For each of these scores, a score of 1 corresponds to always predicting the correct label, while a score of 0 corresponds to no mechanistic understanding of the network.

## Prompting for testing
Test prompting takes place in 2 stages. The first is getting the network to make a prediction. The second is getting this into a format which we can easily process.  

The first prompt includes the background prompt, the layer summaries and the overall summary.
It then uses the following to get the best results:

*Now, thinking step by step, predict what the network will say the following image is:* 

The response is then retrieved and a second prompt is used to get the actual answer in numerical form:  
*Here is a text giving an answer to the question 'What will this MNIST neural network predict for this image?'*  
*[The response from ChatGPT]*  
*What is the answer given? Please respond with a single digit only.*  

## Testing results
Here are the results evaluated on a single model with 6 neurons in a single hidden layer:  
Fraction correct when network correct: 0.98 +- 0.014  
Fraction incorrect when network correct: 0.02 +- 0.014  
Fraction correct about network when network incorrect: 0.04 +- 0.02  
Fraction correct about image when network incorrect: 0.75 +- 0.043  
Fraction incorrect about both network and image: 0.21 +- 0.041  

Final correct interpretability score: 0.92 +- 0.007  
Final incorrect interpretability score: 0.014 +- 0.02  
Final interpretability score: 0.467 +- 0.01  

Clearly, this is very effective at interpreting when the network is right, but doesn't understand what is happening when it's wrong.  
Interestingly, this is very sensitive to the bias we put into the prompt: When we add the sentence 'Note that for half of the images you are asked about , the network prediction will be incorrect', the incorrect interpretability score improves dramatically, while the correct one takes a large hit:
Fraction correct when network correct: 0.51 +- 0.05  
Fraction incorrect when network correct: 0.49 +- 0.05  
Fraction correct about network when network incorrect: 0.12 +- 0.032  
Fraction correct about image when network incorrect: 0.43 +- 0.05  
Fraction incorrect about both network and image: 0.45 +- 0.05  

Final correct interpretability score: 0.14 +- 0.04  
Final incorrect interpretability score: 0.068 +- 0.032  
Final interpretability score: 0.104 +- 0.026  

This shows us that the interpretability score is therefore not necessarily a reflection of the quantity of information in the summaries, and more work needs to be done on correctly prompting ChatGPT to get the most out of its testing predictions. Equally it shows us that there is a lot more work to be done before we can confidently say that ChatGPT can interpret the MNIST dataset. I believe that work on making its summaries more precise, as well as future increases in capabilities will likely improve this fact.
