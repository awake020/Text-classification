import torch
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from seq_indexers.seq_indexer_base import SeqIndexerBase


class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def predict(self, texts, embedding_indexer: SeqIndexerBaseEmbeddings, label_indexer: SeqIndexerBase, batch_size):
        lens = len(texts)
        batch_num = (lens + batch_size - 1) // batch_size
        ans = []
        for i in range(batch_num):
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
