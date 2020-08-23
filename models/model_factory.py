from models.MLP import MLP
from models.text_rnn_attn import TextRNNAttn
from models.text_CNN import TextCNN


class ModelFactory(object):
    @staticmethod
    def get_model(config, args, seq_indexer, label_indexer):
        if config['type'] == 'RNN':
            return TextRNNAttn(embedding_alphabet=seq_indexer,
                               gpu=args.gpu,
                               feat_num=label_indexer.__len__(),
                               **config['model'])
        elif config['type'] == 'CNN':
            return TextCNN(embedding_alphabet=seq_indexer,
                           gpu=args.gpu,
                           feat_num=label_indexer.__len__(),
                           **config['model'])
        elif config['type'] == 'MLP':
            return MLP(embedding_alphabet=seq_indexer,
                       gpu=args.gpu,
                       feat_num=label_indexer.__len__(),
                       **config['model'])
        else:
            raise RuntimeError('no model')
