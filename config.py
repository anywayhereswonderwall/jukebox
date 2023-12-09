import torch

from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 413
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.03
    bias: bool = False

