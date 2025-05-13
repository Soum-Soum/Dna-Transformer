import hashlib
import polars as pl
from tqdm.rich import tqdm
from adn.data.datasets.base import DNADataset
from adn.utils.paths_utils import PathHelper


import pandas as pd
from loguru import logger

REFERENCE_STR = "reference"


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
        logger.info(f"Saving pairs to file {self.cache_save_path}")
        self.cache_save_path.parent.mkdir(parents=True, exist_ok=True)
        pairs_df = pd.DataFrame(pairs, columns=["individual", "snp"])
        pairs_df.to_csv(self.cache_save_path, index=False)
        logger.info(f"Saved pairs to {self.cache_save_path}")

    def _load_pairs(self) -> list[tuple[str, int]]:
        logger.info(f"Loading pairs from file {self.cache_save_path}")
        pairs_df = pd.read_csv(self.cache_save_path)
        pairs = list(zip(pairs_df["individual"], pairs_df["snp"]))
        logger.info(f"Loaded pairs from {self.cache_save_path}")
        return pairs

    def get_individuals_pos_pairs(self) -> list[tuple[str, int]]:

        if self.cache_save_path.exists():
            return self._load_pairs()
        else:
            logger.info(
                f"Cache file {self.cache_save_path} does not exist. Generating pairs..."
            )

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
            pairs.extend(list(zip([REFERENCE_STR] * len(real_seq_id), real_seq_id)))

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

    def _extract_individual_subsequence(
        self, individual: str, snp_idx: int
    ) -> pl.DataFrame:
        if individual != REFERENCE_STR:
            return super()._extract_individual_subsequence(
                individual=individual, snp_idx=snp_idx
            )

        sub_ref = self.reference_genome[snp_idx : snp_idx + self.sequence_length]
        return sub_ref.with_columns(pl.col("main_allele").alias("allele"))

    def get_label(self, individual: str) -> int:
        if individual == REFERENCE_STR:
            return 0
        return super().get_label(individual)

    def __hash__(self):
        individuals = sorted(self.metadata_df.reset_index()["individual"].to_list())
        individual_str = "_".join(individuals)
        s = f"{individual_str}|{self.sequence_length}|{self.overlaping_ratio}"
        return int(hashlib.md5(s.encode()).hexdigest(), 16)

    def __len__(self) -> int:
        return len(self.individual_pos_pairs)

    def __getitem__(self, idx):
        individual, snp_idx = self.individual_pos_pairs[idx]
        sub_ref_updated = self._extract_individual_subsequence(individual, snp_idx)
        return self._subsequence_to_dict(sub_ref_updated, individual)
