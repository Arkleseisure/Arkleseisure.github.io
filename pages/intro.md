---
layout: default
title: Home
---

# Introduction

AI misalignment is an existential threat. Crucial to combatting this potential future terror is the field of mechanistic interpretability, which seeks to understand how neural networks think. However, not much research seems to be directed at fully end-to-end understanding neural networks, to the extent that one might be able to predict these networks outputs based solely on this understanding. Neither has using the power of LLMs to automate this. A notable exception is [this OpenAI paper](https://openai.com/index/language-models-can-explain-neurons-in-language-models/), which attempts to evaluate all of the neurons in GPT2 using GPT4. However, this work notably stopped short of evaluating the relationships between these neurons.

Here, we attempt to fix this gap by fully evaluating networks with ChatGPT 4o, starting with the very simple XOR problem, and attempt to use its vision capabilities to help it evaluate MNIST neural networks. We also introduce an interpretability score to evaluate how well the full network has been understood and summarised. This score varies from 0 to 1, with 0 being no understanding of the network and 1 being full understanding.

Now, while it is probably unfeasible to fully evaluate a network the size of GPT4 for instance, we believe that if in future we are able to scale these techniques to larger sizes, even by losing some information, automated interpretability is going to be fundamental to scaling this field to properly understanding state of the art networks. We believe that answers to big problems are often solved by looking at simpler ones, and hope that this work is a step in the correct direction.
