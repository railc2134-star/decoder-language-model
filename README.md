Char Transformer Decoder

This project is a simple character-level language model built from scratch using PyTorch.

It learns to predict the next character in a sequence using a Transformer-style decoder architecture.

Features

- Character-level vocabulary built from dataset
- Custom tokenization (no pretrained tokenizer)
- Transformer decoder with multi-head self-attention
- Positional embeddings
- Autoregressive text generation
- Trained using CrossEntropy loss

Model Architecture

The model is a decoder-only Transformer composed of:

- Character embedding layer
- Learned positional embeddings
- Multi-head self-attention with causal mask
- Feedforward network with ReLU and dropout
- Layer normalization
- Output projection to vocabulary size

Training

The model is trained to predict the next character in a sequence using teacher forcing.

Input sequence is split into fixed-length chunks of 64 characters.

Loss function is CrossEntropyLoss.

Generation

Text is generated autoregressively:
- Start with a prompt
- Predict next character
- Append it to the input
- Repeat until EOS token or max length

Notes

This is a minimal implementation and not optimized for production use.
It is intended for educational purposes to understand how transformer decoders work at character level.
