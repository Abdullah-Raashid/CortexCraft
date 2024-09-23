import torch # handles the calculus, linear algebra etc.
import torch.nn as nn
import torch.nn.functional as F
import random
import mmap
import pickle
import argparse

parser = argparse.ArgumentParser(description= 'This is a test.')

# Here we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument('-batch_size', type = str, required=True, help = 'Please provide a batch size')

args = parser.parse_args()

# Now we can use the argument value in our program
print(f'Batch size: {args.batch_size}')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(device)



block_size = 128
batch_size = int(args.batch_size)
max_iters= 200
eval_iters = 100
learning_rate = 3e-4
n_embd = 400 # number of dimensions we want to capture
n_head = 1 # numbers of head in parallel
n_layer = 1 # number of decoders
dropout = 0.2

with open('openwebtext/vocab.txt', 'r', encoding='utf-8') as file:
    text = file.read()
# making a vocabulary list to store all the characters

chars = sorted(set(text))
vocab_size = len(chars) # how many unique characters there are

# Tokenizer
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# encode_hello = encode('hello')
# decode_hello = decode(encode_hello)
# print(decode_hello)

def get_random_chunk(split):
    filename = "openwebtext/train_split.txt" if split == 'train' else "openwebtext/test_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)
            
            #seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)
            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors = 'ignore').replace('\r', '')
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype = torch.long)
    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    X = torch.stack([data[i:i+block_size] for i in ix]) # For each index in ix, it extracts a chunk of block_size characters (a sequence) from the data.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # This is used to predict the next character in a sequence during training.
    X,y = X.to(device), y.to(device)
    return X,y


@torch.no_grad() # making sure model is not using any gradients
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,y = get_batch(split)
            logits, loss = model(X,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__ (self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        # Registering no look ahead in the mask state
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # scaled dot product attention
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # computer attention scores ("affinities")
        wei = q @ k.transpose (-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device)).type_as(wei)  # (T, T)
        wei = wei.masked_fill(mask == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
        # masked_fill:
        # [1, -inf, -inf]
        # [1, 0.6, -inf]
        # [1, 0.6, 0.4] then a softmax to score better in the long run

class MultiHeadAttention(nn.Module):
    # multiple heads of self attention in parallel
    def __init__ (self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2...]
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    # simple linear layer followed by nonlinearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout), # preventing overfitting
        )
    def forward (self, x):
        return self.net(x)

class Block(nn.Module):
    # each decoder block
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head # number of features each head is capturing
        self.sa = MultiHeadAttention(n_head, head_size) # sa is self attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        y = self.sa(x) # self attention
        x = self.ln1(x+y) # add and norm
        y = self.ffwd(x) # feedforward
        x = self.ln2(x+y) # add and norm
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # giant grid for predictions, high probability of i coming after an r, 
        # adding the embeddings and POS encoding from the paper
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.blocks is the number of decoder blocks running sequentially, this is creation of the decoders
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)]) 
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm, adding at the end of network for model to converge better
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    # helps converge better
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)

    # Writing a forward pass function from scatch, best practice
    def forward(self, index, targets = None):
        B, T = index.shape

        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device = device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size )
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
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

model = GPTLanguageModel(vocab_size)

model.to(device)

 # Create a pytorch optimzer
optimizer = torch.optim.AdamW(model.parameters(), lr =learning_rate)

for iter in range(max_iters):
    if iter%eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, test loss: {losses['test']:.4f}")
    # Sample a batch of data
    xb, yb = get_batch('train')
    #evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')