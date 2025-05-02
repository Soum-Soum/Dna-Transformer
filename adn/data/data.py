from typing import Optional
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from adn.data.datasets.RandomFixedLenDNADataset import RandomFixedLenDNADataset
from adn.data.datasets.SequentialFixedLenDNADataset import SequentialFixedLenDNADataset
from adn.data.datasets.base import (
    DNADataset,
)
from adn.utils.paths_utils import PathHelper


def load_metadata(
    path_helper: PathHelper,
    labels_to_remove: Optional[str],
    data_ratio_to_use: float,
    individual_to_ignore: Optional[str],
    train_eval_split: float,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:

    metadata = pd.read_csv(path_helper.metadata_file_path)
    logger.info(f"Loaded metadata with {len(metadata)} individuals")

    if labels_to_remove:
        all_labels = set(metadata["label"].unique())
        labels_to_remove = set(labels_to_remove.split(","))
        label_to_use = all_labels - labels_to_remove
        logger.info(f"Using labels: {label_to_use}. Excluding: {labels_to_remove}")
        logger.info(f"Filtering metadata to use only labels: {label_to_use}")
        metadata = metadata[metadata["label"].isin(label_to_use)]

    if individual_to_ignore:
        individuals_to_ignore = load_individuals_to_ignore(individual_to_ignore)
        metadata = metadata[~metadata["individual"].isin(individuals_to_ignore)]
        logger.info(f"Ignoring individuals: {individuals_to_ignore}")

    if data_ratio_to_use < 1.0:
        logger.info(
            f"Using {data_ratio_to_use * 100}% of the data. Original size: {len(metadata)}"
        )
        metadata = metadata.sample(frac=data_ratio_to_use, random_state=42)

    metadata = metadata.set_index("individual")

    if train_eval_split != 0:
        logger.info(
            f"Splitting metadata (len: {len(metadata)}) into train and test with ratio: {train_eval_split}"
        )
        train_metadata, test_metadata, _, _ = train_test_split(
            metadata,
            metadata,
            test_size=train_eval_split,
            random_state=42,
            stratify=metadata["label"],
        )
    else:
        logger.info("Train test split set to 0, using all data for training")
        train_metadata = metadata
        test_metadata = None

    return train_metadata, test_metadata


def load_individuals_to_ignore(individuals_to_ignore: str) -> set[str]:
    with open(individuals_to_ignore, "r") as f:
        individuals = set(f.read().splitlines())
    return individuals


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
    individual_to_ignore: Optional[str] = None,
) -> tuple["DNADataset", Optional["DNADataset"]]:
    train_metadata, test_metadata = load_metadata(
        path_helper=path_helper,
        labels_to_remove=labels_to_remove,
        data_ratio_to_use=data_ratio_to_use,
        individual_to_ignore=individual_to_ignore,
        train_eval_split=train_eval_split,
    )

    label_to_id = {
        label: idx for idx, label in enumerate(train_metadata["label"].unique())
    }

    kwargs = {
        "path_helper": path_helper,
        "label_to_id": label_to_id,
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

    train_dataset = selected_ds_class(metadata_df=train_metadata, **kwargs)
    logger.info(
        f"Train dataset loaded with {len(train_dataset.metadata_df)} individuals"
    )

    if test_metadata is None:
        return train_dataset, None

    test_dataset = selected_ds_class(metadata_df=test_metadata, **kwargs)
    logger.info(f"Test dataset loaded with {len(test_dataset.metadata_df)} individuals")

    return train_dataset, test_dataset
