from adn.data.datasets.base import DNADataset
from adn.utils.paths_utils import PathHelper


import pandas as pd
from loguru import logger


class SequentialFixedLenDNADataset(DNADataset):

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        path_helper: PathHelper,
        label_to_id: dict[str, int],
        sequence_length: int,
        overlaping_ratio: float,
    ):
        super().__init__(
            metadata_df=metadata_df,
            path_helper=path_helper,
            label_to_id=label_to_id,
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

        snp_idx = self.current_position
        self.current_position += int(self.sequence_length * (1 - self.overlaping_ratio))
        return self.get_sequence_dict(
            individual=self.current_individual, snp_idx=snp_idx
        )
