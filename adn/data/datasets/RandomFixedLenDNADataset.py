from adn.data.datasets.base import DNADataset
from adn.utils.paths_utils import PathHelper


import pandas as pd


import random


class RandomFixedLenDNADataset(DNADataset):

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        path_helper: PathHelper,
        label_to_id: dict[str, int],
        sequence_length: int,
        sequence_per_individual: int,
    ):
        super().__init__(
            metadata_df=metadata_df,
            path_helper=path_helper,
            label_to_id=label_to_id,
            sequence_length=sequence_length,
        )
        self.sequence_per_individual = sequence_per_individual

    def __len__(self):
        return len(self.individuals) * self.sequence_per_individual

    def __getitem__(self, index: int) -> dict:
        individual = random.choice(self.individuals)
        snp_idx = random.randint(
            0, self.reference_genome.shape[0] - self.sequence_length
        )
        return self.get_sequence_dict(individual, snp_idx)
