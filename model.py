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
