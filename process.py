import os
from data_io.data_SST_2 import DataIOSST2
from evaluate.eval_factory import EvalFactory
from alphabet.alphabet import Alphabet
from alphabet.alphabet_embedding import AlphabetEmbeddings
import torch
from models.model_factory import ModelFactory
from tqdm import tqdm


class Process(object):
    def __init__(self, config, args, dataset, model, seq_alphabet, label_alphabet):
        self.config = config
        self.args = args
        self.dataset = dataset
        self.train_loader = self.dataset.get_train_loader()
        self.seq_alphabet = seq_alphabet
        self.label_alphabet = label_alphabet
        self.model = model
        self.optimizer = self.get_optimizer(config['hparas'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = EvalFactory().get_eval(config['eval'])



    def get_optimizer(self, config):
        if config['optimizer'] == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        else:
            raise Exception('暂不支持')

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model.predict(self.dataset.train_word,
                                      embedding_alphabet=self.seq_alphabet,
                                      label_alphabet=self.label_alphabet,
                                      batch_size=self.config['data']['batch_size'])
            train_score = self.evaluator.get_score(pred, self.dataset.train_label)

            pred = self.model.predict(self.dataset.dev_word,
                                      embedding_alphabet=self.seq_alphabet,
                                      label_alphabet=self.label_alphabet,
                                      batch_size=self.config['data']['batch_size'])
            dev_score = self.evaluator.get_score(pred, self.dataset.dev_label)

            pred = self.model.predict(self.dataset.test_word,
                                      embedding_alphabet=self.seq_alphabet,
                                      label_alphabet=self.label_alphabet,
                                      batch_size=self.config['data']['batch_size'])
            test_score = self.evaluator.get_score(pred, self.dataset.test_label)

            print('eval train / dev / test | %1.4f / %1.4f / %1.4f.' % (train_score, dev_score, test_score))
            return train_score, dev_score, test_score

    def train(self):
        _, best_score, _ = self.eval()
        early_num = self.config['hparas']['early_num']
        early_stop = self.config['hparas']['early_stop']
        count = 0
        epoch_num = self.config['hparas']['epoch_num']
        for epoch in range(epoch_num):
            count += 1
            self.model.train()
            train_loss = 0.0
            for x, y in tqdm(self.train_loader):
                padded_text, lens, mask = self.seq_alphabet.add_padding_tensor(x, gpu=self.args.gpu)
                label = self.label_alphabet.instance2tensor(y, gpu=self.args.gpu)
                y = self.model(padded_text, lens, mask)
                loss = self.criterion(y, label)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                self.optimizer.zero_grad()

            print('epoch', epoch + 1, ' loss: ',
                  train_loss / (len(self.dataset.train_word) / self.config['data']['batch_size']))

            _, dev_score, _ = self.eval()
            if dev_score > best_score:
                print("saving")
                torch.save(self.model, os.path.join(self.config['save']['path'],
                                                    self.config['type'] + '_' +
                                                    self.config['save']['version'] + '.hdf5'))
                best_score = dev_score
                count = 0
            if early_stop and count >= early_num:
                print('early stop')
                break
