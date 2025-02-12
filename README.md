# AI and Machine Learning Projects

This repository contains several machine learning and artificial intelligence projects, each focusing on different techniques and challenges. Below is an overview of each folder and its purpose.

## Project Structure

### 1. Cipher - Solving the Hardest Problem from the Polish AI Olympiad

In this project, Word2Vec is utilized to tackle one of the most challenging tasks from the 1st Polish Artificial Intelligence Olympiad. The approach leverages natural language processing (NLP) techniques to find meaningful vector representations of words, enabling the model to decipher complex patterns and relationships in text data.

### 2. CIFAR-10 - Training CNN and ResNet18 for Image Classification

This project focuses on training convolutional neural networks (CNNs) on the CIFAR-10 dataset. The implementation includes:

- A simple CNN model.
- A ResNet18 architecture achieving nearly **80% accuracy** within approximately 30 minutes of training on a single CPU.

This result is quite efficient given the hardware limitations, demonstrating the effectiveness of deep learning techniques even with constrained computational resources.

### 3. MCTS - Connect Four with AlphaZero and Standard Monte Carlo Tree Search

This folder contains Python scripts implementing **Monte Carlo Tree Search (MCTS)** for playing the classic **Connect Four** game. The implementation includes:

- A traditional MCTS-based agent.
- An **AlphaZero-inspired** approach leveraging reinforcement learning to improve gameplay decision-making over time.

### 4. Shakespeare - Text Generation using LSTM and GPT

This project involves generating Shakespearean-style text using deep learning models. The dataset consists of **400,000 Shakespearean lines**, and the implementation includes:

- **LSTM-based text generation**.
- **A GPT-inspired model**, following the approach introduced by **Andrej Karpathy** in his famous [YouTube lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY\&t=1s).

This project showcases how deep learning can be applied to natural language generation, mimicking the style of historical texts.

## Getting Started

To explore any of these projects, simply clone this repository:

```bash
git clone git@github.com:yourusername/your-repo-name.git
cd your-repo-name
```

## Contributions

If you’d like to contribute to this repository, feel free to open a pull request or raise an issue for discussion.

## Acknowledgments

- The **Shakespeare GPT implementation** follows the method outlined by **Andrej Karpathy** in his YouTube lecture on GPT from scratch.
- The **MCTS and AlphaZero** methodology is inspired by reinforcement learning research in game AI.

---

This repository serves as an exploration of different AI methodologies, showcasing applications in NLP, computer vision, and game AI.

