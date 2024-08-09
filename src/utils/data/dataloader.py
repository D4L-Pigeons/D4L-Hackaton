from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import Tensor
from typing import Dict
from argparse import Namespace
from utils.data.dataset import hdf5SparseDataset


def _hdf5_custom_collate_fn(batch: Dict[str, Tensor]):
    r"""
    Custom collate function for batching data.
    Applied to the output of the __getitems__ method of the dataset object.

    Args:
        batch (Dict[str, Tensor]): A dictionary containing tensors representing a batch of data.

    Returns:
        Dict[str, Tensor]: The input batch itself, as it does not need to be collated.

    """
    return batch  # Returned batch does not need to be collated


def get_hdf5SparseDataloader(cfg: Namespace, dataset: hdf5SparseDataset):
    r"""
    Returns a DataLoader for loading data from an hdf5SparseDataset.

    Args:
        cfg (Namespace): The configuration object containing the necessary parameters.
            - dataloader.batch_size: The batch_size DataLoader argument.
            - dataloader.num_workers: The num_workers DataLoader argument.
            - dataloader.random: Wether to use RandomSampler of SequentialSampler - relevant for train/validation setups.

    Returns:
        DataLoader: The DataLoader object for loading the data.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        sampler=(
            RandomSampler(dataset)
            if cfg.dataloader.random
            else SequentialSampler(dataset)
        ),
        num_workers=cfg.dataloader.num_workers,
        collate_fn=_hdf5_custom_collate_fn,
        pin_memory=True,
    )
    return dataloader
