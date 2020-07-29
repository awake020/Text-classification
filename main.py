import torch
import json
import random
import argparse
import numpy as np
import os
from data_io.data_SST_2 import DataIOSST2
from models.text_CNN import TextCNN
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from seq_indexers.seq_indexer_base import SeqIndexerBase
from models.MLP import MLP
from tqdm import tqdm
from evaluate.sst_2_f1 import sst2F1Eval

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument("--data_dir", "-dd", type=str, default='data/SST-2')
parser.add_argument("--save_dir", '-sd', type=str, default='save')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--embedding_dir', "-ed", type=str, default='data/glove/glove.6B.100d.txt')
parser.add_argument('--embedding_dim', type=int, default=100)
parser.add_argument('--random_state', '-rs', type=int, default=0)
parser.add_argument('--num_epoch', '-ne', type=int, default=100)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, choices=['CNN', 'MLP'])
args = parser.parse_args()
# model parameters

if __name__ == "__main__":

    # Fix the random seed of package random.
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Fix the random seed of Pytorch when using GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
        torch.cuda.manual_seed(args.random_state)

    # Fix the random seed of Pytorch when using CPU.
    torch.manual_seed(args.random_state)
    torch.random.manual_seed(args.random_state)

    dataset = DataIOSST2(args)
    datasets = dataset.get_data_loader()
    train_loader, dev_loader, test_loader = datasets['train'], datasets['dev'], datasets['test']
    seq_indexer = SeqIndexerBaseEmbeddings("glove", args.embedding_dir, args.embedding_dim, ' ')
    seq_indexer.load_embeddings_from_file()

    label_indexer = SeqIndexerBase("laebl", False, False)
    label_indexer.add_instance(dataset.train_label)

    if args.load is not None:
        model = torch.load(args.load)
        if args.gpu >= 0:
            model.cuda(device=args.gpu)
    else:
        if args.model == 'MLP':
            model = MLP(embedding_indexer=seq_indexer,
                        gpu=args.gpu,
                        feat_num=label_indexer.__len__(),
                        dropout=args.dropout_rate)
        elif args.model == 'CNN':
            model = TextCNN(embedding_indexer=seq_indexer,
                            gpu=args.gpu,
                            feat_num=label_indexer.__len__(),
                            dropout=args.dropout_rate,
                            kernel_size=[2, 3, 5])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)
    eval = sst2F1Eval()
    best_score = 0.0
    count = 0
    for epoch in range(args.num_epoch):
        train_loss = 0.0
        k = 0
        for x, y in tqdm(train_loader):
            padded_text, lens, mask = seq_indexer.add_padding_tensor(x, gpu=args.gpu)
            label = label_indexer.instance2tensor(y, gpu=args.gpu)
            y = model(padded_text, lens, mask)
            loss = criterion(y, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            optimizer.zero_grad()

        print('epoch', epoch + 1, ' loss: ', train_loss / (len(dataset.train_word) / args.batch_size))

        pred = model.predict(dataset.train_word,
                             embedding_indexer=seq_indexer,
                             label_indexer=label_indexer,
                             batch_size=args.batch_size)
        train_score = eval.get_socre(pred, dataset.train_label)

        pred = model.predict(dataset.dev_word,
                             embedding_indexer=seq_indexer,
                             label_indexer=label_indexer,
                             batch_size=args.batch_size)
        dev_score = eval.get_socre(pred, dataset.dev_label)

        pred = model.predict(dataset.test_word, embedding_indexer=seq_indexer, label_indexer=label_indexer,
                             batch_size=args.batch_size)
        test_score = eval.get_socre(pred, dataset.test_label)

        print('eval train / dev / test | %1.2f / %1.2f / %1.2f.' % (train_score, dev_score, test_score))

        if dev_score > best_score:
            model.cpu()
            torch.save(model, os.path.join(args.save_dir, args.model + '.hdf5'))
            if args.gpu >= 0:
                model.cuda(device=args.gpu)
            print('best model saved')
            best_score = dev_score
            count = 0
        else:
            count += 1
            if count >= 5:
                print('already ok')
                break
