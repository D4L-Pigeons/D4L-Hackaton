import numpy as np
import torch

from src.train import train
from src.validate import validate


def main(args):
    torch.manual_seed(100)
    np.random.seed(100)

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        validate(args)
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented yet.")
