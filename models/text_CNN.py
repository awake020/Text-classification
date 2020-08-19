import torch
import torch.nn.functional as F
import torch.nn as nn
from models.basic_model import BasicModel
from alphabet.alphabet_embedding import AlphabetEmbeddings
from models.layers import LayerWordEmbeddings


class TextCNN(BasicModel):
    def __init__(self, embedding_indexer: AlphabetEmbeddings,
                 gpu,
                 feat_num,
                 dropout,
                 kernel_size,
                 fc_dim,
                 cnn_channel=50):
        super(TextCNN, self).__init__()
        self.embedding = LayerWordEmbeddings(embedding_indexer)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, cnn_channel, [x, embedding_indexer.emb_dim])
             for x in kernel_size])
        fc_layers = []
        pre_dim = len(kernel_size) * cnn_channel
        for dim in fc_dim:
            fc_layers.append(nn.Linear(pre_dim, dim))
            pre_dim = dim
        fc_layers.append(nn.Linear(pre_dim, feat_num))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.dropout = nn.Dropout(dropout)
        self.gpu = gpu
        if gpu >= 0:
            self.cuda(device=gpu)

    def conv_and_pool(self, conv, input):
        input = F.relu(conv(input)).squeeze(-1)
        input = F.max_pool1d(input, input.size(-1)).squeeze(-1)
        return input

    def forward(self, input:torch.Tensor, lens, mask):
        input = input.unsqueeze(dim=1)
        input = self.dropout(self.embedding(input))
        out = torch.cat([self.conv_and_pool(conv, input) for conv in self.convs], dim=1)
        first = True
        for layer in self.fc_layers:
            if first == False:
                out = F.relu(out)
            first = False
            out = layer(self.dropout(out))
        return out
