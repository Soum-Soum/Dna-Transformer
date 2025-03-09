from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
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
    sequence_per_individual: int,
    sequence_length: int,
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
    train_dataset = DNADataset(
        metadata_df=train_metadata,
        dataframes=train_dataframes,
        max_position=max_position,
        sequence_per_individual=sequence_per_individual,
        sequence_length=sequence_length,
        label_to_id=label_to_id,
        tokenizer=dna_tokenizer,
    )

    test_dataset = DNADataset(
        metadata_df=test_metadata,
        dataframes=test_dataframes,
        max_position=max_position,
        sequence_per_individual=sequence_per_individual,
        sequence_length=sequence_length,
        label_to_id=label_to_id,
        tokenizer=dna_tokenizer,
    )

    return train_dataset, test_dataset


@dataclass
class DNACursor:
    individual: str
    position: int


class DNADataset(Dataset):

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        dataframes: dict[str, pl.DataFrame],
        max_position: int,
        sequence_per_individual: int,
        sequence_length: int,
        label_to_id: dict[str, int],
        tokenizer: PreTrainedTokenizerFast,
        mode="random",
    ):
        super().__init__()
        self.metadata_df = metadata_df
        self.individuals = self.metadata_df.index.to_list()
        self.dataframes = dataframes
        self.max_position = max_position
        self.sequence_per_individual = sequence_per_individual
        self.sequence_length = sequence_length
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer
        self.mode = mode
        self.cursor = DNACursor(individual=self.individuals[0], position=0)
        logger.info(f"Loaded {len(self.individuals)} individuals")

    def get_random_sub_df(self) -> tuple[pl.DataFrame, str]:
        individual = np.random.choice(list(self.individuals))
        df = self.dataframes[individual]
        snp_idx = np.random.choice(df.shape[0] - self.sequence_length)
        sub_df = df[snp_idx : snp_idx + self.sequence_length]
        return sub_df, individual

    def get_next_sub_df(self) -> tuple[pl.DataFrame, str]:
        df = self.dataframes[self.cursor.individual]
        if self.cursor.position + self.sequence_length >= df.shape[0]:
            self.cursor = DNACursor(
                individual=self.individuals[
                    self.individuals.index(self.cursor.individual) + 1
                ],
                position=0,
            )
            df = self.dataframes[self.cursor.individual]

        sub_df = df[self.cursor.position : self.cursor.position + self.sequence_length]
        return sub_df, self.cursor.individual

    def __len__(self):
        return len(self.individuals) * self.sequence_per_individual

    def __getitem__(self, idx):
        if self.mode == "random":
            sub_df, individual = self.get_random_sub_df()
        elif self.mode == "sequential":
            sub_df, individual = self.get_next_sub_df()
        else:
            raise ValueError(f"Invalid mode : {self.mode}")

        sequence_position = sub_df["position"].to_numpy().astype(np.float32)
        sequence_position = (sequence_position / self.max_position).tolist()
        sequence_position = (
            [sequence_position[0]] + sequence_position + [sequence_position[-1]]
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
        return sequence, label_id, sequence_position
