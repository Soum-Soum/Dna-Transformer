from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast

from adn.tokenizer import get_tokenizer


def load_dataframes(
    individuals_snp_dir: Path, individuals: list[str]
) -> dict[str, pl.DataFrame]:
    snp_parquet_files = list(
        filter(
            lambda x: x.stem in individuals,
            individuals_snp_dir.glob("*.parquet"),
        )
    )
    iterrable = tqdm(snp_parquet_files, desc="Loading SNP data...")
    dataframes = {file.stem: pl.read_parquet(file) for file in iterrable}
    return dataframes


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, sep="\t")
    metadata = metadata[metadata["GroupK4"].isin(["XI", "GJ", "cA"])]
    return metadata


def compute_max_position(dataframes: dict[str, pl.DataFrame]) -> int:
    max_position = 0
    for _, df in dataframes.items():
        max_position = max(max_position, df["position"].max())
    return max_position


def load_datasets(
    individuals_snp_dir: Path,
    metadata_path: Path,
    train_eval_split: float,
    sequence_length: int,
    mode: str,
    sequence_per_individual: int = -1,
    data_ratio_to_use: float = 1.0,
):
    metadata = load_metadata(metadata_path).set_index("individual")
    metadata = metadata.sample(frac=data_ratio_to_use, random_state=42)
    individuals = metadata.index.to_list()
    dataframes = load_dataframes(individuals_snp_dir, individuals)
    max_position = compute_max_position(dataframes)
    train_metadata, test_metadata, _, _ = train_test_split(
        metadata,
        metadata,
        test_size=train_eval_split,
        random_state=42,
        stratify=metadata["GroupK4"],
    )

    label_to_id = {
        label: idx for idx, label in enumerate(train_metadata["GroupK4"].unique())
    }
    dna_tokenizer = get_tokenizer()

    train_dataframes = {
        individual: dataframes[individual] for individual in train_metadata.index
    }
    test_dataframes = {
        individual: dataframes[individual] for individual in test_metadata.index
    }

    kwargs = {
        "max_position": max_position,
        "sequence_length": sequence_length,
        "label_to_id": label_to_id,
        "tokenizer": dna_tokenizer,
    }

    if mode == "random":
        selected_ds_class = RandomDNADataset
        kwargs["sequence_per_individual"] = sequence_per_individual
    elif mode == "sequential":
        selected_ds_class = SequentialDNADataset
    else:
        raise ValueError(f"Unknown mode: {mode}, expected 'random' or 'sequential'")

    train_dataset = selected_ds_class(
        metadata_df=train_metadata, dataframes=train_dataframes, **kwargs
    )

    test_dataset = selected_ds_class(
        metadata_df=test_metadata, dataframes=test_dataframes, **kwargs
    )

    return train_dataset, test_dataset


class DNADataset(Dataset):

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        dataframes: dict[str, pl.DataFrame],
        max_position: int,
        sequence_length: int,
        label_to_id: dict[str, int],
        tokenizer: PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.metadata_df = metadata_df
        self.individuals = sorted(self.metadata_df.index.to_list())
        self.dataframes = dataframes
        self.max_position = max_position
        self.sequence_length = sequence_length
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer

    @property
    def class_weights(self) -> np.ndarray:
        return compute_class_weight(
            "balanced",
            classes=np.array(list(self.label_to_id.keys())),
            y=self.metadata_df["GroupK4"],
        )

    def _prepare_sequence(self, sub_df, individual):
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


class RandomDNADataset(DNADataset):

    def __init__(
        self,
        metadata_df,
        dataframes,
        max_position,
        sequence_per_individual,
        sequence_length,
        label_to_id,
        tokenizer,
    ):
        super().__init__(
            metadata_df,
            dataframes,
            max_position,
            sequence_length,
            label_to_id,
            tokenizer,
        )
        self.sequence_per_individual = sequence_per_individual

    def __len__(self):
        return len(self.individuals) * self.sequence_per_individual

    def __getitem__(self, idx):
        individual = np.random.choice(self.individuals)
        df = self.dataframes[individual]
        snp_idx = np.random.choice(df.shape[0] - self.sequence_length)
        sub_df = df[snp_idx : snp_idx + self.sequence_length]

        return self._prepare_sequence(sub_df, individual)


class SequentialDNADataset(DNADataset):

    def __init__(
        self,
        metadata_df,
        dataframes,
        max_position,
        sequence_length,
        label_to_id,
        tokenizer,
    ):
        super().__init__(
            metadata_df,
            dataframes,
            max_position,
            sequence_length,
            label_to_id,
            tokenizer,
        )
        self.current_individual = self.individuals[0]
        self.current_position = 0

    def __len__(self):
        count = 0
        for individual in self.individuals:
            count += self.dataframes[individual].shape[0] // self.sequence_length
        return count

    def __getitem__(self, idx):
        df = self.dataframes[self.current_individual]
        if self.current_position + self.sequence_length >= df.shape[0]:
            next_individual_idx = self.individuals.index(self.current_individual) + 1

            if next_individual_idx >= len(self.individuals):
                raise StopIteration("End of dataset reached")

            self.current_individual = self.individuals[next_individual_idx]
            self.current_position = 0
            logger.info(
                f"Switching to individual {next_individual_idx} : {self.current_individual}"
            )

            df = self.dataframes[self.current_individual]

        sub_df = df[
            self.current_position : self.current_position + self.sequence_length
        ]
        self.current_position += self.sequence_length
        return self._prepare_sequence(sub_df, self.current_individual)


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
