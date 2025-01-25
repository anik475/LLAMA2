from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    @staticmethod
    def build(checkpoints_dir:str, tokenizer_path:str,load_model:bool, max_seq_len:int, max_batch_size:int, device:str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            chk_path = checkpoints[0]
            assert len(checkpoints) > 0, f'No checkpoints found in {checkpoints_dir}'
            print(f'Loading checkpoints {chk_path}')
            checkpoint = torch.load(chk_path, map_location='cpu')
            print(f'Loaded in {time.time()-prev_time:.2f}s')
            prev_time = time.time()

        with open(Path(checkpoints_dir)/'params.json', 'r') as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Storage)
        
        model = Transformer(model_args).to(device)


        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoints,strict=True)
            print(f'State Dict loaded in {time.time()-prev_time:.2f}s')

        return LLaMA(model, tokenizer, model_args)
    

if __name__ == '__main__':
    torch.manual_seed(0)
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )
    print('Model loaded successfully')