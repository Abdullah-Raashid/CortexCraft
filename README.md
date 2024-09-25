CortexCraft is a prototype language model developed using PyTorch, designed to generate text based on input prompts. The model employs a GPT (Generative Pre-trained Transformer)-like architecture that leverages transformer blocks, self-attention mechanisms, positional encoding, and an autoregressive training approach to generate meaningful sequences of text. The project encapsulates a range of modern deep learning techniques, including tokenization, attention, feed-forward layers, and custom optimization strategies, making it a sophisticated example of a transformer-based language model.

Initial Setup and Hyperparameters
The project begins by importing necessary libraries like torch and its submodules (nn, F) to handle the computational aspects of deep learning. The script then initializes device handling to utilize either a Metal Performance Shader (MPS) or fallback to a CPU, ensuring hardware acceleration when available.

Key hyperparameters are defined:

block_size: 128, determining the sequence length for input/output.
batch_size: 64, specifying how many sequences are processed per iteration.
learning_rate: 3e-4, controlling the gradient step size during optimization.
n_embd: 400, setting the number of embedding dimensions (features).
n_head and n_layer: 1, determining the number of self-attention heads and decoder layers, making this a simplified transformer model.
Tokenization and Vocabulary Construction
The code reads in a text file (vocab.txt from OpenWebText) to create a vocabulary of unique characters for the model to work with. Two dictionaries (string_to_int and int_to_string) are constructed to encode characters into integers and decode them back into text. This character-level encoding allows the model to process text as sequences of individual characters, giving it the flexibility to generate any text output.

Functions encode and decode convert between text and integer representations, which are later used for both training and inference.

Data Loading and Batching
To train the model on realistic text sequences, the code uses a custom data loading mechanism. The get_random_chunk function reads random chunks of text from either the training or test split of the dataset, returning a batch of encoded data. It makes use of memory-mapped files (mmap) to efficiently access large datasets, which avoids the need to load the entire dataset into memory at once.

The get_batch function further processes the data by selecting random starting indices for each batch and creating two tensors (X and y), where X holds the input sequences and y holds the corresponding target sequences (shifted by one character). This arrangement is typical in autoregressive models, where the task is to predict the next token in a sequence.

Model Architecture
CortexCraft is modeled as a decoder-only transformer, akin to the original GPT design. The core components include:

Self-Attention Mechanism: The self-attention mechanism is implemented in the Head and MultiHeadAttention classes. The attention mechanism computes a weighted average of input features, allowing the model to focus on relevant parts of the input when generating each token.

Head defines the key, query, and value projections for attention, where the scaled dot-product attention is computed. It also masks future tokens to ensure that predictions are made in an autoregressive manner.
MultiHeadAttention allows the model to compute attention over multiple subspaces (heads) in parallel, capturing more complex relationships in the input sequence.
Feed-Forward Network: The FeedForward class implements a simple two-layer neural network with ReLU activation and dropout. Each transformer block contains one feed-forward layer, which processes information after self-attention. The network expands the input dimensionality and then reduces it back to ensure rich feature extraction while regularizing with dropout to prevent overfitting.

Transformer Blocks: The Block class represents a single transformer block, which sequentially applies multi-head self-attention and feed-forward layers. Each block is equipped with residual connections (x + y) and layer normalization to improve gradient flow and model convergence. The use of multiple blocks (though simplified here with just 1 layer) allows the model to refine its understanding of the sequence at various levels of abstraction.

Positional Encoding and Embedding: Since transformers have no inherent notion of token order, positional encoding is essential. The model uses learned embeddings for both token positions (position_embedding_table) and characters (token_embedding_table). These embeddings are summed together to provide the model with both positional and semantic information.

Language Model Head: The GPTLanguageModel class encapsulates the entire architecture. After processing the input through multiple blocks, the model uses a linear layer (lm_head) to map the final hidden states to vocabulary logits, which represent probabilities over the next character in the sequence.

Loss Calculation and Training
During the forward pass, the model computes predictions (logits) for each input sequence. If target sequences are provided, the logits are reshaped and cross-entropy loss is calculated, which is the standard loss function for classification problems like language modeling.

The estimate_loss function is used to evaluate the model's performance on both training and test data. By running the model in evaluation mode, it computes average losses over a series of batches, providing a measure of how well the model is generalizing.

Text Generation
The model includes a generate function that allows it to generate text autoregressively. Starting with an initial context (often just a start token or a few characters), it predicts the next character, appends it to the input, and repeats the process for a specified number of tokens. The generation process leverages softmax probabilities and random sampling (torch.multinomial) to choose the next character, allowing the model to produce creative, diverse outputs.

Training Loop and Optimization
The training loop uses the AdamW optimizer, which is well-suited for transformer models due to its decoupled weight decay. The loop iterates over max_iters steps, periodically printing training and test loss for monitoring progress. The optimizer applies gradient-based updates to the model’s parameters, and dropout is used in multiple places to prevent overfitting.

At each iteration:

A batch of data is sampled using get_batch.
The model computes logits and loss.
Gradients are calculated via backpropagation (loss.backward()).
The optimizer steps to update model weights (optimizer.step()).
Model Saving and Loading
The model is periodically saved using Python’s pickle module, allowing it to be reloaded and trained or used later. This provides a mechanism for continuing training or testing without starting from scratch, making it a practical solution for large-scale models and datasets.
