import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iter = 1000
eval_interval = 200
learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embed = 96
n_heads = 3
n_layers = 3
dropout = 0.2
# choose device
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

torch.manual_seed(69)

# read the file (mahabharata)
with open('mahabharata.txt', 'r', encoding='utf-8') as f:
  text = f.read()


# Remove the BOM character if it exists
text = text.replace('\ufeff', '')


# unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)


# code to encode and decode vocabulary
token_to_int = {token:int for int,token in enumerate(chars)}
int_to_token = {int:token for int,token in enumerate(chars)}
encode = lambda s: [token_to_int[c] for c in s] # convert string to list of integers
decode = lambda l: ''.join([int_to_token[i] for i in l]) # convert list of integers to string


# encode the entire text
data = torch.tensor(encode(text), dtype=torch.long)

# train and validation split
n = int(0.9*len(text))
train_data = data[:n]
val_data = data[n:]


# loading data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # won't call .backward() in this function: makes pytorch efficient with memory usage
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module): # single head of self-attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # scaled-attention
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # weighted aggregation
        v = self.value(x)
        out = wei@v

        return out


class MultiHeadAttention(nn.Module):
    # running multiple heads of self-attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # linear projection to residual highway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concat along channel dimension
        out = self.dropout(self.proj(out)) # applying linear projection
        return out


class FeedForward(nn.Module):
    # simple MLP with a ReLU nonlinearity

    def __init__(self, n_embed):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), # as per the paper
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed), # same linear projection, just added to the Sequential
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)
    

class Block(nn.Module):
    # transformer block: intersperse communication and computation
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed//n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.sa(self.ln1(x)) # with residual connections & LayerNorm
        x = x + self.ffn(self.ln2(x)) # with residual connections & LayerNorm
        return x


class NGramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_embed)
        self.pos_emb_table = nn.Embedding(block_size, n_embed)
        # self.sa_head = Head(n_embed)
        # self.sa_heads = MultiHeadAttention(4, n_embed//4) # multi-headed attention with 4 heads, each divided 32 embed dims into 4 = 32/4 = 8-dim self-attention
        # self.ffn = FeedForward(n_embed) # the feed-forward mlp
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are of size (B, T) (batch, time) (4, 8) in this case
        token_emb = self.token_emb_table(idx) # B x T x C, C = n_embed
        pos_emb = self.pos_emb_table(torch.arange(T, device=device)) # T x C
        x = token_emb + pos_emb
        # x = self.sa_head(x) # apply one head of attention
        # x = self.sa_heads(x) # appply multi-head attention
        # x = self.ffn(x) # BxTxC
        x = self.blocks(x) # BxTxT, the attention+mlp blocks
        logits = self.lm_head(x) # B x T x vocab_size


        if targets is None:
            loss=None
        else:
            # cross_entropy expects C as second dim, so we modify the dim of our tensors:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # C as the second dim
            targets = targets.view(B*T) # make this one dimensional
            loss = F.cross_entropy(logits, targets) # should work now

        return logits, loss
    
        # copied this from karpathy's github (easy code)
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to last block_size tokens
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = NGramLanguageModel()
m = model.to(device)


# TRAINING 

# create an optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train') # get a batch of data

    # evaluate
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # to prevent gradient accumulation
    loss.backward()
    optimizer.step()

# infer from the model
context = torch.zeros((1, 1), dtype=torch.long,device=device)

print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))