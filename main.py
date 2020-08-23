import torch
import random
import argparse
import numpy as np

from alphabet.alphabet import Alphabet
from alphabet.alphabet_embedding import AlphabetEmbeddings
from data_io.data_SST_2 import DataIOSST2
from models.model_factory import ModelFactory
from process import Process
import yaml

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument("--config", '-cf', type=str)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--random_state', '-rs', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)

# model parameters

if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
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

    # get dataset and alphabets
    dataset = DataIOSST2(config['data'])
    seq_alphabet = AlphabetEmbeddings(**config['embedding'])
    seq_alphabet.load_embeddings_from_file()
    label_alphabet = Alphabet('label', False, False)
    label_alphabet.add_instance(dataset.train_label)

    # get model
    if args.load is not None:
        model = torch.load(args.load)
        if args.gpu >= 0:
            model.cuda(device=args.gpu)
    else:
        model = ModelFactory.get_model(config, args, seq_alphabet, label_alphabet)

    process = Process(config, args, dataset, model, seq_alphabet, label_alphabet)
    process.train()
