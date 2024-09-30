---
layout: default
title: Home
---

# Introduction

AI misalignment is an existential threat. Crucial to combatting this potential future terror is the field of mechanistic interpretability, which seeks to understand how neural networks think. However, not much research seems to be directed at fully understanding neural networks, or using the power of LLMs to automate this. A notable exception is [this OpenAI paper](https://openai.com/index/language-models-can-explain-neurons-in-language-models/), which attempts to evaluate all of the neurons in GPT2 using GPT4. However, this work notably stopped short of evaluating the relationships between these neurons.

Here, we attempt to fix this gap by fully evaluating networks with ChatGPT 4o, using it to fully evaluate networks for the very simple XOR problem, and attempt to use its vision capabilities to help it evaluate MNIST neural networks. We also introduce an interpretability score to evaluate how well the network has been understood and summarised. This score varies from 0 to 1 and accounts for biases introduced by both the network and ChatGPT being most likely to predict the correct answer to any image.

Now, while it is probably unfeasible to fully evaluate a network the size of GPT4 for instance, we believe that if in future we are able to scale these techniques to larger sizes, even by losing some information, automated interpretability is going to be fundamental to scaling this field to properly understanding state of the art networks. We believe that answers to big problems are often solved by looking at simpler ones, and hope that this work is a step in the correct direction.
