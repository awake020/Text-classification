import torch
import torch.nn.functional as F
import torch.nn as nn
from models.basic_model import BasicModel
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from layers.layer_word_embeddings import LayerWordEmbeddings
from seq_indexers.seq_indexer_base import SeqIndexerBase

class MLP(BasicModel):
    def __init__(self, embedding_indexer: SeqIndexerBaseEmbeddings, gpu, feat_num, dropout, fc_dim):
        super(MLP, self).__init__()
        self.embedding = LayerWordEmbeddings(embedding_indexer)
        self.dropout = torch.nn.Dropout(dropout)
        fc_layers = []
        pre_dim = embedding_indexer.emb_dim
        for dim in fc_dim:
            fc_layers.append(nn.Linear(pre_dim, dim))
            pre_dim = dim
        fc_layers.append(nn.Linear(pre_dim, feat_num))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.gpu = gpu
        if(gpu >=0):
            self.cuda(device=gpu)

    def forward(self, words, lens:torch.Tensor, mask:torch.Tensor):
        words = self.embedding(words)
        words = words * (mask.unsqueeze(-1).expand_as(words))
        words = self.dropout(words)
        out = torch.sum(words, dim=1, keepdim=False) / lens.unsqueeze(-1)
        first = True
        for layer in self.fc_layers:
            if first == False:
                out = F.relu(out)
            first = False
            out = layer(self.dropout(out))
        return out
