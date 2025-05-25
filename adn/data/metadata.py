from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import compute_class_weight
from adn.utils.paths_utils import PathHelper

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


@dataclass
class Metadata:
    metadata_df: pd.DataFrame
    individuals: list[str]
    label_to_id: dict[str, int]
    family_to_id: dict[str, int]
    id_to_label: dict[int, str]
    id_to_family: dict[int, str]
    label_to_family: dict[str, str]
    class_weights: np.ndarray
    family_class_weights: np.ndarray

    @property
    def label_id_to_family_id(self) -> dict[int, int]:
        return {
            self.label_to_id[label]: self.family_to_id[family]
            for label, family in self.label_to_family.items()
        }

    def __len__(self) -> int:
        return len(self.metadata_df)


def split_metadata(
    metadata: Metadata,
    train_eval_split: float,
) -> tuple[Metadata, Optional[Metadata]]:
    if train_eval_split != 0:
        logger.info(
            f"Splitting metadata (len: {len(metadata)}) into train and test with ratio: {train_eval_split}"
        )
        train_metadata_df, test_metadata_df, _, _ = train_test_split(
            metadata.metadata_df,
            metadata.metadata_df,
            test_size=train_eval_split,
            random_state=42,
            stratify=metadata.metadata_df["label"],
        )
    else:
        logger.info("Train test split set to 0, using all data for training")
        train_metadata_df = metadata.metadata_df
        test_metadata_df = None

    def create_meta(df: pd.DataFrame) -> Metadata:
        return Metadata(
            metadata_df=df,
            individuals=sorted(df.index.to_list()),
            label_to_id=metadata.label_to_id,
            family_to_id=metadata.family_to_id,
            id_to_label=metadata.id_to_label,
            id_to_family=metadata.id_to_family,
            label_to_family=metadata.label_to_family,
            class_weights=metadata.class_weights,
            family_class_weights=metadata.family_class_weights,
        )

    train_metadata = create_meta(train_metadata_df)
    test_metadata = (
        create_meta(test_metadata_df) if test_metadata_df is not None else None
    )
    return train_metadata, test_metadata


def load_individuals_to_ignore(individuals_to_ignore: str) -> set[str]:
    with open(individuals_to_ignore, "r") as f:
        individuals = set(f.read().splitlines())
    return individuals


def build_metadata(
    path_helper: PathHelper,
    labels_to_remove: Optional[str],
    data_ratio_to_use: float,
    individuals_to_ignore: Optional[str],
) -> Metadata:
    metadata = pd.read_csv(path_helper.metadata_file_path)
    logger.info(f"Loaded metadata with {len(metadata)} individuals")

    if labels_to_remove:
        all_labels = set(metadata["label"].unique())
        labels_to_remove = set(labels_to_remove.split(","))
        label_to_use = all_labels - labels_to_remove
        logger.info(f"Using labels: {label_to_use}. Excluding: {labels_to_remove}")
        metadata = metadata[metadata["label"].isin(label_to_use)]

    if individuals_to_ignore:
        individuals_to_ignore = load_individuals_to_ignore(individuals_to_ignore)
        metadata = metadata[~metadata["individual"].isin(individuals_to_ignore)]
        logger.info(f"Ignoring individuals: {individuals_to_ignore}")

    if data_ratio_to_use < 1.0:
        logger.info(
            f"Using {data_ratio_to_use * 100}% of the data. Original size: {len(metadata)}"
        )
        metadata = metadata.sample(frac=data_ratio_to_use, random_state=42)

    metadata = metadata.set_index("individual")

    individuals = sorted(metadata.index.to_list())
    label_to_id = {
        label: idx for idx, label in enumerate(metadata["label"].sort_values().unique())
    }
    family_to_id = {
        family: idx
        for idx, family in enumerate(metadata["family"].sort_values().unique())
    }
    id_to_label = {v: k for k, v in label_to_id.items()}
    id_to_family = {v: k for k, v in family_to_id.items()}
    label_to_family = metadata.groupby(metadata["label"])["family"].first().to_dict()

    # Calculate class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array(list(label_to_id.keys())),
        y=metadata["label"],
    )

    family_class_weights = compute_class_weight(
        "balanced",
        classes=np.array(list(family_to_id.keys())),
        y=metadata["family"],
    )

    return Metadata(
        metadata_df=metadata,
        individuals=individuals,
        label_to_id=label_to_id,
        family_to_id=family_to_id,
        id_to_label=id_to_label,
        id_to_family=id_to_family,
        label_to_family=label_to_family,
        class_weights=class_weights,
        family_class_weights=family_class_weights,
    )
