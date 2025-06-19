from typing import Optional

from loguru import logger

from adn.data.datasets.RandomFixedLenDNADataset import RandomFixedLenDNADataset
from adn.data.datasets.SequentialFixedLenDNADataset2 import SequentialFixedLenDNADataset
from adn.data.datasets.base import (
    DNADataset,
)
from adn.data.metadata import build_metadata, split_metadata
from adn.utils.paths_utils import PathHelper


class DatasetMode:
    RANDOM_FIXED_LEN = "random_fixed_len"
    SEQUENTIAL_FIXED_LEN = "sequential_fixed_len"


def load_datasets(
    path_helper: PathHelper,
    train_eval_split: float,
    sequence_length: int,
    mode: DatasetMode,
    sequence_per_individual: int = -1,
    overlaping_ratio: float = -1,
    data_ratio_to_use: float = 1.0,
    labels_to_remove: Optional[str] = None,
    individuals_to_ignore: Optional[str] = None,
) -> tuple["DNADataset", Optional["DNADataset"]]:

    metadata = build_metadata(
        metadata_file_path=path_helper.metadata_file_path,
        labels_to_remove=labels_to_remove,
        data_ratio_to_use=data_ratio_to_use,
        individuals_to_ignore=individuals_to_ignore,
    )

    train_metadata, test_metadata = split_metadata(
        metadata=metadata,
        train_eval_split=train_eval_split,
    )

    kwargs = {
        "path_helper": path_helper,
        "sequence_length": sequence_length,
    }

    if mode == DatasetMode.RANDOM_FIXED_LEN:
        assert (
            sequence_per_individual > 0
        ), "Sequence per individual must be greater than 0"
        selected_ds_class = RandomFixedLenDNADataset
        kwargs["sequence_per_individual"] = sequence_per_individual
    elif mode == DatasetMode.SEQUENTIAL_FIXED_LEN:
        assert (
            sequence_per_individual == -1
        ), "Sequence per individual not supported in sequential mode"
        selected_ds_class = SequentialFixedLenDNADataset
        kwargs["overlaping_ratio"] = overlaping_ratio
    else:
        raise ValueError(f"Unknown mode: {mode}, expected 'random' or 'sequential'")

    train_dataset = selected_ds_class(metadata=train_metadata, **kwargs)
    logger.info(
        f"Train dataset loaded with {len(train_dataset.metadata)} individuals. Ds len : {len(train_dataset)}"
    )

    if test_metadata is None:
        return train_dataset, None

    test_dataset = selected_ds_class(metadata=test_metadata, **kwargs)
    logger.info(
        f"Test dataset loaded with {len(test_dataset.metadata)} individuals. Ds len : {len(test_dataset)}"
    )

    return train_dataset, test_dataset
