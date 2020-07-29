import torch
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from seq_indexers.seq_indexer_base import SeqIndexerBase


class Basic_Model(torch.nn.Module):
    def __init__(self):
        super(Basic_Model, self).__init__()

