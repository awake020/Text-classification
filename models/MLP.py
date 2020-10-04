import torch
import torch.nn.functional as F
import torch.nn as nn
from models.basic_model import BasicModel, WordEmbeddings
from alphabet.alphabet_embedding import AlphabetEmbeddings


class MLP(BasicModel):
    def __init__(self, embedding_alphabet: AlphabetEmbeddings, gpu, feat_num, dropout, fc_dim):
        super(MLP, self).__init__()
        self.embedding = WordEmbeddings(embedding_alphabet)
        self.dropout = torch.nn.Dropout(dropout)
        fc_layers = []
        pre_dim = embedding_alphabet.emb_dim
        for dim in fc_dim:
            fc_layers.append(nn.Linear(pre_dim, dim))
            pre_dim = dim
        self.fc_layers = nn.ModuleList(fc_layers)
        self.final_layer = nn.Linear(pre_dim, feat_num)

        self.gpu = gpu

    def forward(self, words, lens: torch.Tensor, mask: torch.Tensor):
        words = self.embedding(words)
        words = words * (mask.unsqueeze(-1).expand_as(words))
        words = self.dropout(words)
        out = torch.sum(words, dim=1, keepdim=False) / lens.unsqueeze(-1)
        for layer in self.fc_layers:
            out = layer(self.dropout(out))
            out = F.relu(out)
        out = self.final_layer(self.dropout(out))
        return out
