import polars as pl
from tqdm.rich import tqdm
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
        self.cache_save_path = path_helper.ds_cache_dir / f"pairs_{hash(self)}.csv"
        self.individual_pos_pairs = self.get_individuals_pos_pairs()

    def _save_pairs(self, pairs: list[tuple[str, int]]):
        logger.info("Saving pairs to file")
        self.cache_save_path.parent.mkdir(parents=True, exist_ok=True)
        pairs_df = pd.DataFrame(pairs, columns=["individual", "snp"])
        pairs_df.to_csv(self.cache_save_path, index=False)
        logger.info(f"Saved pairs to {self.cache_save_path}")

    def _load_pairs(self) -> list[tuple[str, int]]:
        logger.info("Loading pairs from file")
        pairs_df = pd.read_csv(self.cache_save_path)
        pairs = list(zip(pairs_df["individual"], pairs_df["snp"]))
        logger.info(f"Loaded pairs from {self.cache_save_path}")
        return pairs

    def get_individuals_pos_pairs(self) -> list[tuple[str, int]]:

        if self.cache_save_path.exists():
            return self._load_pairs()

        offsets = list(
            range(
                0,
                self.sequence_length,
                int(round(self.sequence_length * self.overlaping_ratio)),
            )
        )

        reference = self.reference_genome

        pairs = []

        for offset in offsets:
            real_seq_id = list(
                range(
                    offset,
                    self.reference_genome.shape[0] - offset,
                    self.sequence_length,
                )
            )
            pairs.extend(list(zip(["reference"] * len(real_seq_id), real_seq_id)))

        for individual in tqdm(
            self.individuals, desc="Processing pairs...", unit="individual"
        ):
            indiv_df = self.snp_per_individual[individual]
            joined = reference.join(
                indiv_df[["allele", "position"]], on="position", how="left"
            )

            if "__index_level_0__" in joined.columns:
                joined = joined.drop("__index_level_0__")

            for offset in offsets:
                sub_df = joined.slice(offset, None)

                sub_df = sub_df.with_columns(
                    pl.arange(0, sub_df.height)
                    .floordiv(self.sequence_length)
                    .alias("seq_id")
                )

                grouped = (
                    sub_df.group_by("seq_id")
                    .agg(pl.col("allele").drop_nulls().first().alias("allele"))
                    .filter(pl.col("allele").is_not_null())
                    .with_columns(
                        (pl.col("seq_id") * self.sequence_length + offset).alias(
                            "real_seq_id"
                        )
                    )
                    .sort("seq_id")
                )

                pairs.extend([(individual, i) for i in grouped["real_seq_id"]])

        self._save_pairs(pairs)
        return pairs

    def __hash__(self):
        return hash(
            (
                tuple(self.metadata_df.reset_index()["individual"]),
                self.sequence_length,
                self.overlaping_ratio,
            )
        )

    def __len__(self) -> int:
        return len(self.individual_pos_pairs)

    def __getitem__(self, idx):
        individual, snp_idx = self.individual_pos_pairs[idx]
        sub_ref_updated = self._extract_individual_subsequence(individual, snp_idx)
        return self._subsequence_to_dict(sub_ref_updated, individual)
