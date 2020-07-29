import torch
import torch.nn.functional as F

from seq_indexers.seq_indexer_base import SeqIndexerBase
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from layers.layer_word_embeddings import LayerWordEmbeddings


class TextCNN(torch.nn.Module) :
    def __init__(self, embedding_indexer: SeqIndexerBaseEmbeddings, gpu, feat_num, dropout, kernel_size, channel=50):
        super(TextCNN, self).__init__()
        self.embedding = LayerWordEmbeddings(embedding_indexer)
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(1, channel, [x, embedding_indexer.emb_dim]) for x in kernel_size])
        self.fc = torch.nn.Linear(len(kernel_size) * channel, feat_num)
        self.dropout = dropout
        self.gpu = gpu
        if gpu >= 0:
            self.cuda(device=gpu)

    def conv_and_pool(self, conv, input):
        input = F.relu(conv(input)).squeeze(-1)
        input = F.max_pool1d(input, input.size(-1)).squeeze(-1)
        return input

    def forward(self, input:torch.Tensor, lens, mask):
        input = input.unsqueeze(dim=1)
        input = self.embedding(input)
        after_conv = torch.cat([self.conv_and_pool(conv, input) for conv in self.convs], dim=1)
        out = F.dropout(after_conv, p=self.dropout)
        out = self.fc(out)
        return out

    def predict(self, texts, embedding_indexer: SeqIndexerBaseEmbeddings, label_indexer:SeqIndexerBase, batch_size):
        lens = len(texts)
        batch_num = (lens + batch_size - 1) // batch_size
        ans = []
        for i in range(batch_num) :
            start = i * batch_size
            end = min(start + batch_size, lens)
            part = texts[start:end]
            part, lengths, mask = embedding_indexer.add_padding_tensor(part, gpu=self.gpu)
            pred = self.forward(part, lengths, mask)
            pred = torch.argmax(pred, dim=-1, keepdim=False)
            pred = pred.tolist()
            pred = label_indexer.get_instance(pred)
            ans.extend(pred)
        return ans