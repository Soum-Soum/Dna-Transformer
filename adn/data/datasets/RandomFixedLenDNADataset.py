from adn.data.datasets.base import DNADataset
from adn.data.metadata import Metadata
from adn.utils.paths_utils import PathHelper


import pandas as pd


import random


class RandomFixedLenDNADataset(DNADataset):

    def __init__(
        self,
        metadata: Metadata,
        path_helper: PathHelper,
        sequence_length: int,
        sequence_per_individual: int,
    ):
        super().__init__(
            metadata=metadata,
            path_helper=path_helper,
            sequence_length=sequence_length,
        )
        self.sequence_per_individual = sequence_per_individual

    def __len__(self):
        return len(self.metadata.individuals) * self.sequence_per_individual

    def __getitem__(self, index: int) -> dict:
        individual = random.choice(self.metadata.individuals)
        snp_idx = random.randint(
            0, self.reference_genome.shape[0] - self.sequence_length
        )

        sub_ref_updated = self._extract_individual_subsequence(individual, snp_idx)
        return self._subsequence_to_dict(sub_ref_updated, individual)
