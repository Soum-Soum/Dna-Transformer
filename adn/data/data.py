from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import Dataset
from tqdm.rich import tqdm
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast

from adn.models.tokenizer import get_tokenizer
from adn.utils.paths_utils import PathHelper

labels = {"XI", "GJ", "cA"}


def load_snp_per_individual(
    path_helper: PathHelper, individuals: list[str]
) -> dict[str, pl.DataFrame]:
    snp_parquet_files = list(
        filter(
            lambda x: x.stem in individuals,
            path_helper.list_snps_per_individual_paths,
        )
    )
    iterrable = tqdm(snp_parquet_files, desc="Loading SNP data...")
    dataframes = {file.stem: pl.read_parquet(file) for file in iterrable}
    return dataframes


def load_metadata(
    path_helper: PathHelper,
    labels_to_remove: Optional[str],
    data_ratio_to_use: float,
    individual_to_ignore: Optional[str],
) -> pd.DataFrame:
    if labels_to_remove:
        labels_to_remove = set(labels_to_remove.split(","))
        label_to_use = labels - labels_to_remove
        logger.info(f"Using labels: {label_to_use}. Excluding: {labels_to_remove}")
    else:
        label_to_use = labels

    metadata = pd.read_csv(path_helper.metadata_file_path, sep="\t")
    metadata = metadata[metadata["GroupK4"].isin(label_to_use)]
    if individual_to_ignore:
        individuals_to_ignore = load_individuals_to_ignore(individual_to_ignore)
        metadata = metadata[~metadata["individual"].isin(individuals_to_ignore)]
        logger.info(f"Ignoring individuals: {individuals_to_ignore}")
    metadata = metadata.sample(frac=data_ratio_to_use, random_state=42)
    return metadata


def load_ref_genome(path_helper: PathHelper) -> pl.DataFrame:
    reference_genome = pl.read_parquet(path_helper.all_main_alleles_file_path)
    reference_genome = reference_genome.sort("position")
    return reference_genome


def compute_max_position(dataframes: dict[str, pl.DataFrame]) -> int:
    max_position = 0
    for _, df in dataframes.items():
        max_position = max(max_position, df["position"].max())
    return max_position


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
    metadata = load_metadata(
        path_helper, labels_to_remove, data_ratio_to_use, individual_to_ignore
    ).set_index("individual")
    individuals = metadata.index.to_list()
    snp_per_individual = load_snp_per_individual(path_helper, individuals)
    max_position = compute_max_position(snp_per_individual)
    reference_genome = load_ref_genome(path_helper)

    if train_eval_split != 0:
        train_metadata, test_metadata, _, _ = train_test_split(
            metadata,
            metadata,
            test_size=train_eval_split,
            random_state=42,
            stratify=metadata["GroupK4"],
        )
    else:
        logger.info("Train test split set to 0, using all data for training")
        train_metadata = metadata
        test_metadata = None

    label_to_id = {
        label: idx for idx, label in enumerate(train_metadata["GroupK4"].unique())
    }
    dna_tokenizer = get_tokenizer()

    kwargs = {
        "max_position": max_position,
        "label_to_id": label_to_id,
        "tokenizer": dna_tokenizer,
        "reference_genome": reference_genome,
    }

    if mode == DatasetMode.RANDOM_FIXED_LEN:
        assert (
            sequence_per_individual > 0
        ), "Sequence per individual must be greater than 0"
        selected_ds_class = RandomFixedLenDNADataset
        kwargs["sequence_per_individual"] = sequence_per_individual
        kwargs["sequence_length"] = sequence_length
    elif mode == DatasetMode.SEQUENTIAL_FIXED_LEN:
        assert (
            sequence_per_individual == -1
        ), "Sequence per individual not supported in sequential mode"
        selected_ds_class = SequentialFixedLenDNADataset
        kwargs["overlaping_ratio"] = overlaping_ratio
        kwargs["sequence_length"] = sequence_length
    else:
        raise ValueError(f"Unknown mode: {mode}, expected 'random' or 'sequential'")

    train_snps = {
        individual: snp_per_individual[individual]
        for individual in train_metadata.index
    }

    train_dataset = selected_ds_class(
        metadata_df=train_metadata, snp_per_individual=train_snps, **kwargs
    )
    logger.info(
        f"Train dataset loaded with {len(train_dataset.metadata_df)} individuals"
    )

    if test_metadata is None:
        return train_dataset, None

    test_snps = {
        individual: snp_per_individual[individual] for individual in test_metadata.index
    }

    test_dataset = selected_ds_class(
        metadata_df=test_metadata, snp_per_individual=test_snps, **kwargs
    )
    logger.info(f"Test dataset loaded with {len(test_dataset.metadata_df)} individuals")

    return train_dataset, test_dataset


def data_collator(features: list) -> dict:
    input_ids = torch.tensor([f["input_ids"] for f in features])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([f["labels"] for f in features])
    chromosome_positions = torch.tensor([f["chromosome_positions"] for f in features])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "chromosome_positions": chromosome_positions,
    }


class DNADataset(Dataset):

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        snp_per_individual: dict[str, pl.DataFrame],
        reference_genome: pl.DataFrame,
        max_position: int,
        label_to_id: dict[str, int],
        tokenizer: PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.metadata_df = metadata_df
        self.individuals = sorted(self.metadata_df.index.to_list())
        self.snp_per_individual = snp_per_individual
        self.reference_genome = reference_genome
        self.max_position = max_position
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer

    @property
    def id_to_label(self) -> dict[int, str]:
        return {v: k for k, v in self.label_to_id.items()}

    @property
    def class_weights(self) -> np.ndarray:
        return compute_class_weight(
            "balanced",
            classes=np.array(list(self.label_to_id.keys())),
            y=self.metadata_df["GroupK4"],
        )

    def _prepare_sequence_v1(self, sub_df: pl.DataFrame, individual: str) -> dict:
        sequence_position = sub_df["position"].to_numpy().astype(np.float32)
        scaled_sequence_position = (sequence_position / self.max_position).tolist()
        # Duplicate start and end positions to match with BOS and EOS tokens
        scaled_sequence_position = (
            [scaled_sequence_position[0]]
            + scaled_sequence_position
            + [scaled_sequence_position[-1]]
        )

        sequence = (
            sub_df[["main_allele", "allele"]]
            .map_rows(lambda x: "".join(x))
            .to_numpy()
            .squeeze()
            .tolist()
        )
        sequence = " ".join(sequence)
        sequence = self.tokenizer.encode(sequence)

        label = self.metadata_df.loc[individual, "GroupK4"]
        label_id = self.label_to_id[label]

        return {
            "input_ids": sequence,
            "labels": label_id,
            "chromosome_positions": scaled_sequence_position,
            "interval": (sequence_position[0], sequence_position[-1]),
            "individual": individual,
        }

    def _prepare_sequence_v2(self, sub_df: pl.DataFrame, individual: str):
        sequence_position = sub_df["position"].to_numpy().astype(np.float32)
        scaled_sequence_position = (sequence_position / self.max_position).tolist()
        # Duplicate start and end positions to match with BOS and EOS tokens
        scaled_sequence_position = (
            [scaled_sequence_position[0]]
            + scaled_sequence_position
            + [scaled_sequence_position[-1]]
        )

        sequence = (
            sub_df[["main_allele", "allele"]]
            .map_rows(lambda x: "".join(x))
            .to_numpy()
            .squeeze()
            .tolist()
        )
        sequence = " ".join(sequence)
        sequence = self.tokenizer.encode(sequence)

        label = self.metadata_df.loc[individual, "GroupK4"]
        label_id = self.label_to_id[label]

        return {
            "input_ids": sequence,
            "labels": label_id,
            "chromosome_positions": scaled_sequence_position,
            "interval": (sequence_position[0], sequence_position[-1]),
            "individual": individual,
        }


class FixedLenDNADataset(DNADataset):

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        dataframes: dict[str, pl.DataFrame],
        max_position: int,
        label_to_id: dict[str, int],
        tokenizer: PreTrainedTokenizerFast,
        sequence_length: int,
    ):
        super().__init__(
            metadata_df=metadata_df,
            snp_per_individual=dataframes,
            max_position=max_position,
            label_to_id=label_to_id,
            tokenizer=tokenizer,
        )
        self.sequence_length = sequence_length


class RandomFixedLenDNADataset(FixedLenDNADataset):

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        dataframes: dict[str, pl.DataFrame],
        max_position: int,
        label_to_id: dict[str, int],
        tokenizer: PreTrainedTokenizerFast,
        sequence_length: int,
        sequence_per_individual: int,
    ):
        super().__init__(
            metadata_df=metadata_df,
            dataframes=dataframes,
            max_position=max_position,
            label_to_id=label_to_id,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
        )
        self.sequence_per_individual = sequence_per_individual

    def __len__(self):
        return len(self.individuals) * self.sequence_per_individual

    def __getitem__(self, idx):
        individual = np.random.choice(self.individuals)
        df = self.snp_per_individual[individual]
        snp_idx = np.random.choice(df.shape[0] - self.sequence_length)
        sub_df = df[snp_idx : snp_idx + self.sequence_length]

        return self._prepare_sequence(sub_df, individual)


class SequentialFixedLenDNADataset(FixedLenDNADataset):

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        dataframes: dict[str, pl.DataFrame],
        max_position: int,
        label_to_id: dict[str, int],
        tokenizer: PreTrainedTokenizerFast,
        sequence_length: int,
        overlaping_ratio: float,
    ):
        super().__init__(
            metadata_df=metadata_df,
            dataframes=dataframes,
            max_position=max_position,
            label_to_id=label_to_id,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
        )
        self.overlaping_ratio = overlaping_ratio
        self.current_individual = self.individuals[0]
        self.current_position = 0

    def __len__(self):
        count = 0
        for individual in self.individuals:
            count += self.snp_per_individual[individual].shape[0] // int(
                self.sequence_length * (1 - self.overlaping_ratio)
            )
        return count

    def __getitem__(self, idx):
        df = self.snp_per_individual[self.current_individual]
        if self.current_position + self.sequence_length >= df.shape[0]:
            next_individual_idx = self.individuals.index(self.current_individual) + 1

            if next_individual_idx >= len(self.individuals):
                raise StopIteration("End of dataset reached")

            self.current_individual = self.individuals[next_individual_idx]
            self.current_position = 0
            logger.info(
                f"Switching to individual {next_individual_idx} : {self.current_individual}"
            )

            df = self.snp_per_individual[self.current_individual]

        sub_df = df[
            self.current_position : self.current_position + self.sequence_length
        ]
        self.current_position += int(self.sequence_length * (1 - self.overlaping_ratio))
        return self._prepare_sequence(sub_df, self.current_individual)
