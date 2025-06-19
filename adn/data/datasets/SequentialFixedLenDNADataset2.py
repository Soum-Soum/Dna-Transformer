import hashlib
import polars as pl
from tqdm.rich import tqdm
from adn.data.datasets.base import DNADataset
from adn.data.metadata import Metadata
from adn.utils.paths_utils import PathHelper


import pandas as pd
from loguru import logger

REFERENCE_STR = "reference"


class SequentialFixedLenDNADataset(DNADataset):

    def __init__(
        self,
        metadata: Metadata,
        path_helper: PathHelper,
        sequence_length: int,
        overlaping_ratio: float,
    ):
        super().__init__(
            metadata=metadata,
            path_helper=path_helper,
            sequence_length=sequence_length,
        )
        self.overlaping_ratio = overlaping_ratio
        self.pairs = self.compute_pairs()

    def compute_pairs(self) -> list[tuple[str, int]]:
        ref_len = len(self.reference_genome) - self.sequence_length
        offset = int(round(self.sequence_length * (1 - self.overlaping_ratio)))
        idx = range(0, ref_len, offset)
        all_pairs = []
        for individual in self.metadata.individuals:
            all_pairs.extend([(individual, i) for i in idx])
        return all_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        individual, snp_idx = self.pairs[idx]
        sub_ref_updated = self._extract_individual_subsequence(individual, snp_idx)
        return self._subsequence_to_dict(sub_ref_updated, individual)
