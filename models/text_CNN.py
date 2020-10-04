import torch
import torch.nn.functional as F
import torch.nn as nn
from models.basic_model import BasicModel, WordEmbeddings
from alphabet.alphabet_embedding import AlphabetEmbeddings


class TextCNN(BasicModel):
    def __init__(self, embedding_alphabet: AlphabetEmbeddings,
                 gpu,
                 feat_num, # 分类种类数
                 dropout, # dropout值
                 kernel_size, # 卷积核大小 list:[2, 3, 5]
                 fc_dim, # 全连接层size list：[100, 50, 25]
                 cnn_channel=50): # 每种卷积核个数
        super(TextCNN, self).__init__()
        self.embedding = WordEmbeddings(embedding_alphabet)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, cnn_channel, [x, embedding_alphabet.emb_dim])
             for x in kernel_size])
        fc_layers = []
        pre_dim = len(kernel_size) * cnn_channel
        for dim in fc_dim:
            fc_layers.append(nn.Linear(pre_dim, dim))
            pre_dim = dim
        self.fc_layers = nn.ModuleList(fc_layers)
        self.final_layer = nn.Linear(pre_dim, feat_num)
        self.dropout = nn.Dropout(dropout)
        self.gpu = gpu


    def conv_and_pool(self, conv, input):
        input = F.relu(conv(input)).squeeze(-1)
        # B × 1 × T × E
        # FILTER： CHANNELS × SIZE × E
        # == B × CHANNELS × （T - SIZE +１）　×　１
        input = F.max_pool1d(input, input.size(-1)).squeeze(-1)
        return input # B * channels

    def forward(self, input:torch.Tensor, lens, mask):
        input = input.unsqueeze(dim=1) # B × T -> B * 1 * T
        input = self.dropout(self.embedding(input)) # B × 1 × T -> B * 1 * T × E
        out = torch.cat([self.conv_and_pool(conv, input) for conv in self.convs], dim=1)
        # B * (n * channels)
        for layer in self.fc_layers:
            out = layer(self.dropout(out))
            out = F.relu(out)
        out = self.final_layer(self.dropout(out))
        return out
