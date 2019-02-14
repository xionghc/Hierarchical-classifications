import torch
import torch.nn as nn
import torch.nn.functional as F


NODE_SIZE = 174
EMBED_DIM = 128
WINDOW_SIZE = 3


class SementicModel(nn.Module):
    def __init__(self, node_size, embedding_dim):
        super(SementicModel, self).__init__()
        self.embed = nn.Embedding(node_size, embedding_dim)

    def l2_norm(self, emb):
        if len(emb.size()) > 1:
            dim = 1
        else:
            dim = 0
        return F.normalize(emb, dim=dim)

    def forward(self, node):
        embedding = self.embed(node)
        normalized_emb = self.l2_norm(embedding)
        return normalized_emb
