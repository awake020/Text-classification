import torch
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from layers.layer_base import LayerBase


class LayerWordEmbeddings(LayerBase):
    def __init__(self, embedding_indexer:SeqIndexerBaseEmbeddings, freeze_word_embeddings=False):
        super(LayerWordEmbeddings, self).__init__()
        self.word_seq_indexer = embedding_indexer
        embedding_tensor = embedding_indexer.get_loaded_embeddings_tensor()
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings=embedding_tensor, freeze=freeze_word_embeddings)

    def forward(self, input_tensor): # shape: batch_size x max_seq_len
        word_embeddings_feature = self.embeddings(input_tensor) # shape: batch_size x max_seq_len x output_dim
        return word_embeddings_feature
