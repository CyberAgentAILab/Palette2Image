import torch
import torch.nn as nn

# fix random seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

config = {
    'vocab': {'[PST]': 0, '[PAD]': 1, '[UNK]': 2, '[SEP]': 3, '[PEND]': 4}, 
    'd_model': 768, # aligin with text encoder of CLIP
    'seq_len': 21,
}

class PaletteRGBEmbedder(nn.Module):
    def __init__(self):
        super(PaletteRGBEmbedder, self).__init__()
        vocab = config['vocab']
        d_model = config['d_model']
        seq_len = config['seq_len']
        self.tok_embed = TokenEmbedding(vocab, d_model)  # token embedding
        self.seg_embed = SegmentEmbedding(seq_len, d_model)  # position embedding
        self.pos_embed = PositionalEmbedding(seq_len, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        token_embedded = self.tok_embed(x)
        segment_embedded = self.seg_embed(x)
        positional_embedded = self.pos_embed(x)
        embedding = token_embedded + segment_embedded + positional_embedded
        return self.norm(embedding)
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab, embedding_dim):
        super(TokenEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.token_tag_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.text_embedding_layer = nn.Embedding(len(vocab), embedding_dim)
        self.number_embedding_layer = nn.Linear(1, embedding_dim)

    def forward(self, sequence):
        final_embeddings = torch.empty(0, 0, self.embedding_dim)
        for seq in sequence:
            tokens = seq.split()
            embeddings = torch.empty(0, 0, self.embedding_dim)
            for token in tokens:
                if token.isdigit():  # for numbers, normalize to 0-1  
                    num = torch.tensor([[[float(token) / 255]]], dtype=torch.float)
                    num_embedding = self.number_embedding_layer(num)
                    if embeddings.nelement() == 0:
                        embeddings = token_embedding
                    else:
                        embeddings = torch.cat([embeddings, num_embedding], dim=1)
                else:  # for tags
                    idx = torch.tensor([[self.token_tag_to_idx[token]]], dtype=torch.long)
                    token_embedding = self.text_embedding_layer(idx)
                    if embeddings.nelement() == 0:
                        embeddings = token_embedding
                    else:
                        embeddings = torch.cat([embeddings, token_embedding], dim=1)
            if final_embeddings.nelement() == 0:
                final_embeddings = embeddings
            else:
                final_embeddings = torch.cat([final_embeddings, embeddings], dim=0)
        return final_embeddings

# [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
class SegmentEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_dim):
        super(SegmentEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.segment_embedding = nn.Embedding(seq_len, embedding_dim)

    def forward(self, sequence):
        batch_size = len(sequence)
        tokens = sequence[0].split()
        embeddings = []
        seg = 0
        for token in tokens:
            embeddings.append(seg)
            if token == "[SEP]":
                seg += 1
        embeddings = torch.tensor(embeddings).unsqueeze(0).repeat(batch_size, 1)
        return self.segment_embedding(embeddings)
    
# [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.positional_embedding = nn.Embedding(seq_len, embedding_dim)

    def forward(self, sequence):
        batch_size = len(sequence)
        tokens = sequence[0].split()
        embeddings = []
        pos = [0, 1, 2, 3]
        p = 0
        for token in tokens:
            if token == "[SEP]" or token == "[PEND]":
                p = 0
            embeddings.append(pos[p])
            p += 1
        embeddings = torch.tensor(embeddings).unsqueeze(0).repeat(batch_size, 1)
        return self.positional_embedding(embeddings)

