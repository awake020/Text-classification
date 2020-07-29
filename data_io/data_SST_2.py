import csv
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy


class DataIOSST2(object):
    def __init__(self, args):
        self.args = args
        self.train_word, self.train_label, \
        self.dev_word, self.dev_label, \
        self.test_word, \
        self.test_label = self.read_train_dev_test()

    def read_train_dev_test(self):
        train_word, train_label = self.read_data(self.args.data_dir + '/train.tsv')
        dev_word, dev_label = self.read_data(self.args.data_dir + '/dev.tsv')
        test_word, test_label = self.read_data(self.args.data_dir + '/test.tsv')
        return train_word, train_label, dev_word, dev_label, test_word, test_label

    @staticmethod
    def read_data(path):
        data = []
        label = []
        csv.register_dialect('my', delimiter='\t', quoting=csv.QUOTE_ALL)
        with open(path) as tsvfile:
            file_list = csv.reader(tsvfile, "my")
            first = True
            for line in file_list:
                if first:
                    first = False
                    continue
                data.append(line[1].strip().split(" "))
                label.append(line[0])
        csv.unregister_dialect('my')
        return data, label

    def get_data_loader(self):
        data_sets = dict()
        data_sets['train'] = DataLoader(TorchDataset(self.train_word, self.train_label),
                                        batch_size=self.args.batch_size,
                                        shuffle=True,
                                        collate_fn=self.__collate_fn)

        data_sets['dev'] = DataLoader(TorchDataset(self.dev_word, self.dev_label),
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      collate_fn=self.__collate_fn)
        data_sets['test'] = DataLoader(TorchDataset(self.test_word, self.test_label),
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       collate_fn=self.__collate_fn)
        return data_sets

    @staticmethod
    def __collate_fn(batch):
        """
        helper function to instantiate a DataLoader Object.
        """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch


class TorchDataset(Dataset):
    def __init__(self, word, label):
        self.word = word
        self.label = label

    def __getitem__(self, item):
        return self.word[item], self.label[item]

    def __len__(self):
        return len(self.word)
