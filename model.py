import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 256
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[int] = None
    norm_eps: float = 1e-5

    # KV Cache settings
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def precompute_theeta_pos_frequencies(head_dim: int, seq_len:int, device:str, theeta: float=10000.0):

    assert head_dim % 2 == 0, "head_dim must be even"
    # build the theeta values ^ (-2(i-1)/dim) for i in [1, dim//2]
    # According the the formula theeta_i: 10000 
    # shape: (head_dim//2)
    theeta_numerator = torch.arrange(0,head_dim,2).float() 
    # shape: (head_dim//2)
    theeta = 1.0 / (theeta ** (theeta_numerator / head_dim)).to(device)
    #construct the positions ( the "m" parameters)
    # shape: (seq_len)
    m =  torch.arange(seq_len,device=device)
    # multiply each theetha with each position using outer product
    # shape: (seq_len) outer (head_dim/2) -> (seq_len, head_dim/2)
    freqs = torch.outer(m,theeta).float()
    # we want to compute complex number using euler's formula in polar form
    # (seq_len, head_dim/2) -> (seq_len, head_dim/2, 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_pos_encodings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # x: (B, seq_len, H, dim) -> (B, seq_len, H, dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, dim/2) * (1, seq_len, 1, head_dim/2) -> (B, seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, head_dim/2) -> (B,seq_len, H, head_dim/2 ,2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim/2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return{
            x[:, :, :,None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads*n_rep, head_dim)
        }

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        #the gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        #(B, seq_len, dim) * (B, seq_len, 1) = (B, seq_len, dim)
        # rsqrt: 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (dim) * (B, seq_len, dim) -> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)
    
class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim /3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * args.dim)
        #round off
        hidden = args.multiple_of * ((hidden + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish*x_v
        x = self.w2(x)
        return x






class EndcoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        #Normalization before Attenstion
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        #Normalization before Feed Forward
        self.ff_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x:torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ff_norm(h))
        return out

class SelfAttention(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        #Indicate the number of Key and values
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        # Indicate the number of heads for the queries
        self.n_heads_q =  args.n_heads
        # Indicate how many times the keys and values are repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicate the dimension of the Each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    #Grouped Querry Attention
    def forward(self,x:torch.Tensor, start_pos:int, freqs_complex:torch.Tensor):
        
        batch_size, seq_len, _ = x.shape #(B,1,Dim)
        # (B, 1, dim) -> (B, 1, H_Q, head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, H_K, head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q, head_dim) -> (B, 1, H_Q, head_dim)
        xq = xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        # (B, 1, H_K, head_dim) -> (B, 1, H_K, head_dim)
        xk = xk.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        # (B, 1, H_V, head_dim) -> (B, 1, H_V, head_dim)
        xv = xv.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        # Apply the rotary position encodings to Q and K 
        xq = apply_rotary_pos_encodings(xq, freqs_complex, x.device)
        xk = apply_rotary_pos_encodings(xk, freqs_complex, x.device)

        #Replace the output of this token with the cache
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # Retrieve the cache
        # (B,seq_len_kv,H_K,head_dim) -> (B, seq_len_kv, H_K, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        #repeat the keys and values
        keys = repeat_kv(keys,self.n_rep)
        values = repeat_kv(values,self.n_rep)

        # (B, 1, H_Q, head_dim) -> (B, H_Q, 1, head_dim)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        # (B, H_Q, 1, head_dim) * (B, H_K, head_dim, seq_len_kv) -> (B, H_Q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # (B, H_Q, 1, seq_len) * (B, H_V, head_dim, seq_len_kv) -> (B, H_Q, head_dim)
        output = torch.matmul(scores, values)

        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len,-1))
        return self.wo(output) 


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size!=-1, "vocab_size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.nn_layers = args.n_layers
        self.tok_embeddibgs = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(args.n_layers):
            self.layers.append(EndcoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output - nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theeta_pos_frequencies(self.args.dim//self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)


    def forward(self, tokens: torch.Tensor, start_pos: int):
        #(batch_size, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "seq_len must be 1"

        #(B,seq_len)  -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        #retrieve the pairs (m,theeta) coreesponding to the position [start_pos, start_pos+seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        #Consicutively apply the layers
        for layer in self.layers:
            h = layer(h,start_pos,freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
