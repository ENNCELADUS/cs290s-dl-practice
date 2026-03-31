# Project 1: Text Classification with Deep Learning

**Project Instruction**  
**Deadline:** 2026-4-12 23:59 (Late submissions will be penalized by 1 point per day)  
**Submission Format:** Final report must be in PDF. Code zipped as: studentID-name-project1.zip  
**Submission Link:**  
- upload code to https://epan.shanghaitech.edu.cn/l/LFiRKQ
- upload final report to https://ecourse.shanghaitech.edu.cn/  

**Dataset:** ./data
**GPU Resources:** https://aistation2.shanghaitech.edu.cn:32206/

## Environment Setup with uv
Use `uv` as the package manager for local development and HPC runs. The locked environment lives in `pyproject.toml` and `uv.lock`, while `requirements.txt` is kept only for compatibility.

```bash
cd projects/project1-text-classification
UV_CACHE_DIR=/tmp/uv-cache uv sync
source .venv/bin/activate
uv run python train.py --config ./configs/lstm_classifier.toml
```

Named experiment configs live in `configs/`, for example:
- `configs/lstm_classifier.toml`
- `configs/bert_classifier.toml`
- `configs/bert_classifier_advanced.toml`

Training logs are written to `runs/<experiment_name>/<timestamped_run>/` and the best
checkpoint is written to `checkpoints/<experiment_name>/best_model.pt`. To inspect a
run, start TensorBoard from the project root:

```bash
tensorboard --logdir ./runs
```

For example, the advanced BERT run `bert_classifier_advanced_20260331-025421` writes
TensorBoard events under
`runs/bert_classifier_advanced/bert_classifier_advanced_20260331-025421/`, including
files such as
`events.out.tfevents.1774896863.richard-Lenovo-ThinkBook-16p-Gen-4.601938.0.bert_classifier_advanced_20260331-025421`.

For transformer experiments, you may pre-download the backbone with Hugging Face CLI:

```bash
hf download hfl/chinese-roberta-wwm-ext --local-dir ./hf_models/chinese-roberta-wwm-ext
```

If the HPC already provides Python 3.11, keep the same major/minor version there. If needed, create the environment explicitly with `UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.11`.

## 1. Introduction
Sentiment analysis is a classic text classification task that aims to determine the emotional polarity (positive/negative) of a given document. Over the past few decades, many neural architectures have been proposed – from simple recurrent and convolutional networks to large pre‑training based models – each offering different trade‑offs between accuracy, speed, and complexity.

This project focuses on exploring different neural network architectures for sentiment analysis. You will implement and compare at least two different models(such as RNN, CNN, or Transformer-based model)to classify text into positive or negative sentiment categories. 

Through this project, you will gain insight into how different architectures capture linguistic features and contextual information. You will also learn to analyze model performance using appropriate metrics.

## 2. Basic Task
Your primary goal is to implement two different sentiment classification models and compare their performance. Suggested model architectures include RNN-based models (LSTM/GRU), CNN-based models and Transformer-based models (e.g., BERT/GPT). 
**The two models must be different types of architectures.**

### Steps:
1. **Dataset Implementation**
- Implement a custom PyTorch Dataset class in `data.py`.
- use `./data/train.csv` for training and `./data/val.csv` for validation. 

2. **Model Implementation**
- Implement your models in `models/`.  
- You may use PyTorch’s built-in layers (nn.LSTM, nn.Conv1d, nn.Embedding) and/or Hugging Face transformers library for pretrained models. 
- Do not simply copy-paste complete implementations without understanding – the code should be your own work, and you must be able to explain every component.

3. **Training**
- Train both models and monitor training/validation loss and metrics.
- Adjust hyperparameters (e.g., rnn layer size, convolutional kernel size, learning rate) to optimize model performance.
- Report metrics: accuracy, F1-score, precision, recall, and other relevant metrics you find useful.
- Save the best model checkpoint.
- Save the training logs (e.g., loss, accuracy, F1-score) for both models.

4. **Analysis**
- Compare the two architectures in terms of performance.
- Your report should include final performance scores and a discussion of which model performed better and why.

**Expected result:** Both models should achieve reasonable performance on the validation set:  
**accuracy > 0.92** or **macro F1-score > 0.86**.  


## 3. Advanced Tasks
### Model Architecture Optimization
Select one of your two models from the Basic Task and perform architecture optimization to improve its performance – without simply adjusting hyperparameters or changing model size. You must modify the structure of the neural network itself.
**Requirements:**  
- Clearly describe the baseline architecture and your modified version.
- Keep all training hyperparameters identical to isolate the effect of the architectural change.
- Report metrics
- Analyze why the modification helped (or did not help).
- In Week 7 class, share your results and findings.

**Guidelines:**
- For the traditional model (RNN/CNN):
Possible improvements:
  - Incorporate attention mechanisms
  - Combine two architecture into an ensemble or a hybrid architecture (e.g., CNN + LSTM).

- For the pretrained Transformer model (BERT/GPT):
Possible improvements:
  - Enhanced Pooling & Classification Heads
  - Add task‑specific adapters(or LoRA).

## 4. Report Requirements
Your PDF report should include:
1. **Introduction** 
- Briefly describe the problem, the models you chose, and the goal of the project.

2. **Dataset & Preprocessing**  
- Description of the dataset, vocabulary construction, tokenization, and any special preprocessing steps.

3. **Model Architectures**
- For Basic Task: Provide detailed descriptions of both models.  
- For Advanced Task: Clearly present the baseline and the modified architecture. Highlight the exact change you made. 

4. **Training Setup** 
- List the hyperparameters (learning rate, batch size, optimizer, loss function, epochs) used for each model. 
- Ensure you keep hyperparameters identical for the baseline and the improved model in the Advanced Task.

5. **Results with tables and figures**  

6. **Comparative analysis and discussion**  
- Compare the two basic models: strengths, weaknesses, metrics.
- Analyze the effect of your architectural improvement.  

7. **Conclusion and future work**  
- Summarize the key findings from your project.
- Discuss the implications of your results and the potential for future improvements.

8. **References (if any)**

Code should be well-structured, commented and zipped as: studentID-name-project1.zip. Training logs should be included in the zip file.

## 5. Resources
[PyTorch Tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
[Hugging Face Transformers](https://huggingface.co/docs/transformers/models)
Zhou et al., 2016. [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://aclanthology.org/P16-2034/)
Yang et al., 2016. [Hierarchical Attention Networks for Document Classification](https://aclanthology.org/N16-1174/)
Yoon Kim. 2014. [Convolutional Neural Networks for Sentence Classification.](https://arxiv.org/abs/1408.5882)
Devlin et al., 2018. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
Radford et al., 2018. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1)
