import torch
import torch.nn.functional as F
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from layers.layer_word_embeddings import LayerWordEmbeddings
from seq_indexers.seq_indexer_base import SeqIndexerBase

class MLP(torch.nn.Module):
    def __init__(self, embedding_indexer: SeqIndexerBaseEmbeddings, gpu, feat_num, dropout):
        super(MLP, self).__init__()
        self.embedding = LayerWordEmbeddings(embedding_indexer)
        self.linear1 = torch.nn.Linear(embedding_indexer.emb_dim, 50)
        self.linear2 = torch.nn.Linear(50, feat_num)
        self.dropout = torch.nn.Dropout(dropout)
        self.gpu = gpu
        if(gpu >=0):
            self.cuda(device=gpu)

    def forward(self, words, lens:torch.Tensor, mask:torch.Tensor):
        words = self.embedding(words)
        words = words * (mask.unsqueeze(-1).expand_as(words))
        words = self.dropout(words)
        words = torch.sum(words, dim=1, keepdim=False) / lens.unsqueeze(-1)
        words = self.linear1(words)
        words = F.relu(words)
        words = self.dropout(words)
        words = self.linear2(words)
        return words

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
