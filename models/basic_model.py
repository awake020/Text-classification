import torch
from alphabet.alphabet_embedding import AlphabetEmbeddings
from alphabet.alphabet import Alphabet


class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def predict(self, texts, embedding_alphabet: AlphabetEmbeddings, label_alphabet: Alphabet, batch_size):
        lens = len(texts)
        batch_num = (lens + batch_size - 1) // batch_size
        ans = []
        for i in range(batch_num):
            start = i * batch_size
            end = min(start + batch_size, lens)
            part = texts[start:end]
            part, lengths, mask = embedding_alphabet.add_padding_tensor(part, gpu=self.gpu)
            pred = self.forward(part, lengths, mask)
            pred = torch.argmax(pred, dim=-1, keepdim=False)
            pred = pred.tolist()
            pred = label_alphabet.get_instance(pred)
            ans.extend(pred)
        return ans


class WordEmbeddings(torch.nn.Module):
    def __init__(self, alphabet:AlphabetEmbeddings, freeze_word_embeddings=False):
        super(WordEmbeddings, self).__init__()
        self.word_seq_indexer = alphabet
        if self.word_seq_indexer.use_pre_embedding:
            embedding_tensor = alphabet.get_loaded_embeddings_tensor()
            self.embeddings = torch.nn.Embedding.from_pretrained(embeddings=embedding_tensor, freeze=freeze_word_embeddings)
        else:
            self.embeddings = torch.nn.Embedding(len(alphabet), alphabet.emb_dim)

    def forward(self, input_tensor): # shape: batch_size x max_seq_len
        word_embeddings_feature = self.embeddings(input_tensor) # shape: batch_size x max_seq_len x output_dim
        return word_embeddings_feature