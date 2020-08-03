import torch
import random
import argparse
import numpy as np
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
    process = Process(config, args)
    process.train()