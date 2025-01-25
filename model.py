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
