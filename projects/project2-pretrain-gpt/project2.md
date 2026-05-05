# Project 2: Pre-training a GPT Model from Scratch

**Project Instruction**  
**Deadline:** 2026-5-10 23:59 (Late submissions will be penalized by 1 point per day)  
**Submission Format:** Final report must be in PDF. Code zipped as: studentID-name-project2.zip  
**Submission Link:**  
- upload code to https://epan.shanghaitech.edu.cn/l/b1HE3K
- upload final report to https://ecourse.shanghaitech.edu.cn/  

**Dataset Options:** `hongloumeng.txt` or [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) (choose one).
**GPU Resources:** https://aistation2.shanghaitech.edu.cn:32206/

## 1. Introduction

Generative Pre-trained Transformer (GPT) models have revolutionized natural language processing by demonstrating remarkable capabilities in text generation, few-shot learning, and reasoning. Unlike classification models that map input to a label, GPT is an autoregressive language model that learns to predict the next token given the previous context. 

In this project, you will implement a GPT model using PyTorch, without relying on high-level transformer libraries (e.g., Hugging Face Transformers). You will pre-train it on a text corpus (TinyStories or hongloumeng), experiment with different model sizes, and analyze the generated text quality. Finally, you may complete an **advanced task**: either **architectural optimization** (improving some aspect of the base Transformer) or **KV cache implementation** (accelerating inference).

Through this project, you will gain hands-on experience with:
- Building transformer blocks from scratch (e.g., multi-head attention, feed-forward networks).
- Implementing autoregressive language modeling with causal masking.
- Training language models and generating coherent text.
- Practical techniques for optimizing Transformer training or inference.

## 2. Basic Task

Your primary goal is to implement a GPT model from scratch in PyTorch, pre-train it on a chosen dataset, and compare different model sizes.

### Steps:

#### 2.1 Dataset Implementation
- Choose a dataset (hongloumeng.txt or TinyStories). 
   - For TinyStories: Because the complete dataset is very large, you may use only a subset for training. The size of the subset is up to you – choose an amount that fits your GPU memory and time budget. Please clearly state in your report how many samples you used.
   - For hongloumeng: This dataset is relatively small. You may either use it alone or supplement it with other suitable Chinese text data of your choice. If you add extra data, describe it clearly in your report.
- Implement a custom PyTorch `Dataset` class in `dataset.py`.
- Create training and validation splits.

#### 2.2 Model Implementation
- Implement your GPT model in `gpt.py`. **Do not** import pre-built transformer layers from `torch.nn` or Hugging Face.
- Required components:
  - Token embedding layer
  - Positional encoding
  - Multiple transformer decoder blocks, each containing:
    - Multi-head causal self-attention (with masked attention)
    - Feed-forward network
    - Residual connections and normalization
  - Final layer norm and linear head (tied weights optional)
- Implement different model sizes. 

#### 2.3 Training
- Train models on the chosen dataset.
- Monitor the training and validation cross-entropy loss, validation perplexity, and generate sample text during validation.
- Save training logs (loss curves, validation perplexity, sample text).

#### 2.4 Generation and Evaluation
- Implement a text generation function with adjustable temperature and top‑k/top‑p sampling.
- Generate at least 3 text samples from each model using the same prompt.
- Evaluate models qualitatively (coherence, creativity, repetition) and quantitatively (perplexity on validation set).

#### 2.5 Analysis
- Compare different model sizes in terms of:
  - Validation perplexity (lower is better)
  - Training time and memory usage
  - Quality of generated text (provide examples)

## 3. Advanced Tasks (Choose ONE)

Select one of the following two advanced tasks. 

### 3.1 Task A – Architecture Optimization

Improve the **base Transformer architecture** in at least one of the following dimensions: **generation quality, training speed, inference speed, GPU memory usage**, or **other applicable metrics**. 

#### Requirements:

- Choose your largest model as the baseline. 
- The model optimization must be a **structural modification** to the Transformer architecture (not just hyperparameter tuning). 
- Train the optimized model and compare against the baseline on the same validation set. 
- Report metrics: perplexity, training time per epoch, inference time per token, peak memory usage (whichever is relevant to your optimization goal).
- Analyze why the modification helped (or did not help) and provide an explanation in your report.

### 3.2 Task B – KV Cache Implementation

Implement **KV cache** for autoregressive generation and measure its impact on inference speed.

#### Requirements:

- **Modify your largest model** to support KV caching. 
   - The KV cache stores the key and value tensors from previous tokens, so that at each new token, you only compute attention for the new token instead of re‑computing over the entire sequence.
- **Comparison:** Measure the inference time (seconds per generated token or total time to generate a fixed length) **with** and **without** KV cache. Use the same prompt and generation parameters.

## 4. Report Requirements

Your PDF report must include the following sections:

1. **Introduction**  

2. **Dataset & Preprocessing**  
   - Describe the dataset you chose and any additional data you added.
   - Explain your tokenization method and vocabulary size.

3. **Model Architectures**  
   - Describe the GPT architecture you implemented from scratch. 
   - Include the hyperparameter tables for each model sizes.
   - For the Advanced Task A: Clearly present the baseline architecture and the modified version. Highlight the exact structural change.

4. **Training Setup**  
   - List the training hyperparameters: learning rate, batch size, number of epochs, etc.

5. **Results**  
   - Training curves (loss and perplexity) for each model.
   - Sample generated text (at least 3 prompts) from each model.
   - For the Advanced Task: comparison table of relevant metrics (perplexity, training speed, inference speed, memory usage) between baseline and optimized/cached model.

6. **Comparative Analysis and Discussion**  
   - For the Basic Task: compare different model sizes, discuss how increasing capacity affects loss, perplexity, generation quality, and training speed.
   - For the Advanced Task: analyze the impact of your modification – did it achieve the intended improvement? Why or why not?

7. **Conclusion**  
   - Summarize key findings.

8. **References** (if any)

## 5. Code Requirements

- All code must be well-structured, and commented.
- Zip your code as: `studentID-name-project2.zip`
- Do **not** include the dataset and the model checkpoints.

## 6. Resources

- Radford et al., 2018. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1)
- Vaswani et al., 2017. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)