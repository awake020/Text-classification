import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.basic_model import BasicModel, WordEmbeddings
from alphabet.alphabet_embedding import AlphabetEmbeddings


class TextRNNAttn(BasicModel):
    def __init__(self, embedding_alphabet: AlphabetEmbeddings, gpu, feat_num,
                 dropout, hidden_dim, layer_num, fc_dim, bidirectional=True):
        super(TextRNNAttn, self).__init__()
        self.embedding = WordEmbeddings(embedding_alphabet)
        self.gpu = gpu
        self.dropout = nn.Dropout(dropout)
        self.direction = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        if layer_num >= 2:
            self.lstm = nn.LSTM(embedding_alphabet.emb_dim, hidden_dim, layer_num, batch_first=True, dropout=dropout,
                                bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(embedding_alphabet.emb_dim, hidden_dim, layer_num, batch_first=True,
                                bidirectional=bidirectional)
        self.attn = nn.Parameter(torch.randn(self.direction * self.hidden_dim, 1, dtype=torch.float))
        fc_layers = []
        pre_dim = self.direction * self.hidden_dim
        for dim in fc_dim:
            fc_layers.append(nn.Linear(pre_dim, dim))
            pre_dim = dim
        self.fc_layers = nn.ModuleList(fc_layers)
        self.final_layer = nn.Linear(pre_dim, feat_num)


    def sort_by_seq_len_list(self, lens):
        sorted_lens, sorted_indices = torch.sort(lens, descending=True)
        _, reverse_sort_indices = torch.sort(sorted_indices, descending=False)
        return sorted_lens, sorted_indices, reverse_sort_indices

    def pack(self, input_tensor, lens):
        sorted_lens, sort_index, reverse_sort_index = self.sort_by_seq_len_list(lens)
        input = torch.index_select(input_tensor, dim=0, index=sort_index)
        return pack_padded_sequence(input, lengths=sorted_lens, batch_first=True), \
               reverse_sort_index

    def unpack(self, out_pack, reverse_sort_index, lens):
        out_tensor, _ = pad_packed_sequence(out_pack, batch_first=True, total_length=torch.max(lens))
        out_tensor = torch.index_select(out_tensor, index=reverse_sort_index, dim=0)
        return out_tensor

    def generate_hc(self, batch_size):
        h0 = torch.zeros(self.layer_num * self.direction, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.layer_num * self.direction, batch_size, self.hidden_dim)
        if self.gpu >= 0:
            h0 = h0.cuda(0)
            c0 = c0.cuda(0)
        return h0, c0

    def forward(self, inputs: torch.Tensor, lens, mask):
        batch_size = inputs.size(0)
        inputs = self.dropout(self.embedding(inputs))
        inputs, reverse_index = self.pack(inputs, lens)
        h0, c0 = self.generate_hc(batch_size)
        rnn_pack, _ = self.lstm(inputs, (h0, c0))
        rnn_out = self.unpack(rnn_pack, reverse_index, lens)
        a = torch.matmul(rnn_out, self.attn).squeeze(-1)
        a = a + (1 - mask) * -9999999999
        a = F.softmax(a, dim=-1)
        out = rnn_out * (a.unsqueeze(-1))
        out = torch.sum(out, dim=1, keepdim=False) / lens.unsqueeze(-1)

        for layer in self.fc_layers:
            out = layer(self.dropout(out))
            out = F.relu(out)
        out = self.final_layer(self.dropout(out))
        return out
