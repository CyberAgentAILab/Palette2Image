import torch
import torch.nn as nn

config = {
    'vocab_size': 671, 
    'd_model': 768, # aligin with text encoder of CLIP
    'seq_len': 7,
}

# fix random seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

class PaletteEmbedder(nn.Module):
    def __init__(self):
        super(PaletteEmbedder, self).__init__()
        vocab_size = config['vocab_size']
        d_model = config['d_model']
        seq_len = config['seq_len']
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = PositionalEmbedding(seq_len, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        token_embedded = self.tok_embed(x)
        positional_embedded = self.pos_embed(x)
        embedding = token_embedded + positional_embedded
        return self.norm(embedding)

    
class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, hidden_size):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = nn.Embedding(max_len, hidden_size)
        positions = torch.arange(0, max_len)
        self.register_buffer('positions', positions)

    def forward(self, sequence):
        if sequence.dim() == 1: # input a single list[]
            seq_len = sequence.size(0)
            positions = torch.arange(seq_len, dtype=torch.long)
            positions = positions.unsqueeze(0)[0]
        elif sequence.dim() > 1: # input batch list[[]]
            batch_size, seq_len = sequence.size()
            positions = self.positions[:seq_len].unsqueeze(0).repeat(batch_size, 1)
        else:
            return "Empty sequence"

        return self.positional_embedding(positions)