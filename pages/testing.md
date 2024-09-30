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

        prompt = gpt_prompt_base + 'You have evaluated the layers of the network as follows:\n'
        for layer in range(len(layer_summaries)):
            prompt += f'Layer {layer}:\n{layer_summaries[layer]}'
        prompt += 'You summarised the network as working overall as follows:\n'
        prompt += network_summary
        prompt += (
            'Now, thinking step by step, predict what the network will say the following image is. '
            'Ensure that the final character of your response is the digit you would like to answer with, and note that for '
            'half of the images you are asked about, the network prediction will be incorrect.\n'

The response is then retrieved and a second prompt is used to get the actual answer in numerical form:
*Here is a text giving an answer to the question 'What will this MNIST neural network predict for this image?'*  
*[The response from ChatGPT]*  
*What is the answer given? Please respond with a single digit only.*  

## Testing results
Here are the results evaluated on a single model with 6 neurons in the hidden layer:

