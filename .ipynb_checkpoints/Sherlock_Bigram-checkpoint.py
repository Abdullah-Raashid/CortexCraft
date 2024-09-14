#!/usr/bin/env python
# coding: utf-8

# ## Training a Bigram language model using the original Sherlock Holmes novel
Trying to mimic the style of Arthur Conan Doyle
# In[1]:


import torch # handles the calculus, linear algebra etc.
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)
block_size = 8
batch_size =4


# In[2]:


with open('Sherlock_Holmes.txt', 'r', encoding='utf-8') as file:
    text = file.read()
# making a vocabulary list to store all the characters

chars = sorted(set(text))
vocab_size = len(chars) # how many unique characters there are


# In[3]:


# Tokenizer
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# encode_hello = encode('hello')
# decode_hello = decode(encode_hello)
# print(decode_hello)

data = torch.tensor(encode(text), dtype = torch.long) #long sequence of integers
print(data[:100])


# ## Validation and training splits

# In[4]:


n = int(0.8*len(data))
train_data = data[:n]
test_data = data[n:]


# In[5]:


def get_batch(split):
    data = train_data if split =='train' else test_data
    ix = torch.randint(len(data)-block_size,(batch_size,)) # takes a random integer between 1 and end of len(data),represent positions in the text where sequences will be extracted from.
    print(ix) # random indices from the text to start generating from
    X = torch.stack([data[i:i+block_size] for i in ix]) # For each index in ix, it extracts a chunk of block_size characters (a sequence) from the data.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # This is used to predict the next character in a sequence during training.
    X,y = X.to(device), y.to(device)
    return X,y

X,y = get_batch('train')
print('inputs:')
# print(X.shape)
print(X)
print('targets:')
print(y)
    


# In[6]:


x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t] 
    print('when input is ', context, ' target is ', target)


# ## Initializing the neural net

# In[9]:


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # giant grid for predictions, high probability of i coming after an r, 
        # normalize each row to predict what should come after each letter (it should have the highest probability).
    
    # Writing a forward pass function from scatch, best practice
    def forward(self, index, targets = None):
        logits = self.token_embedding_table(index)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # Batch, time, and channels(vocab_size), unpacks them
            logits = logits.view(B*T, C) # reshapes them since it requires (N,C) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :]# becomes (B,C)
            # apply sofmax to get probabilities
            probs = F.softmax(logits, dim = -1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim = 1)
        return index

model = BigramLanguageModel(vocab_size)
m = model.to(device)

context = torch.zeros((1,1), dtype = torch.long, device= device)
generated_chars = decode(m.generate(context, max_new_tokens = 500)[0].tolist())
print(generated_chars)


# In[ ]:




