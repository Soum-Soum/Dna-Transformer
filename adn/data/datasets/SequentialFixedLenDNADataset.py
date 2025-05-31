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
        self.cache_save_path = path_helper.ds_cache_dir / f"pairs_{hash(self)}.parquet"
        self._individual_pos_pairs = self._get_individuals_pos_pairs()
        self._current_individual = None

    def _save_pairs(self, pairs_df: pl.DataFrame):
        logger.info(f"Saving pairs to file {self.cache_save_path}")
        self.cache_save_path.parent.mkdir(parents=True, exist_ok=True)
        pairs_df.write_parquet(self.cache_save_path)
        logger.info(f"Saved pairs to {self.cache_save_path}")

    def _load_pairs(self) -> pl.DataFrame:
        logger.info(f"Loading pairs from file {self.cache_save_path}")
        pairs_df = pl.from_pandas(pd.read_parquet(self.cache_save_path))
        logger.info(f"Loaded pairs from {self.cache_save_path}")
        return pairs_df

    def _get_individuals_pos_pairs(self) -> list[tuple[str, int]]:
        if self.cache_save_path.exists():
            return self._load_pairs()

        logger.info(
            f"Cache file {self.cache_save_path} does not exist. Generating pairs..."
        )

        offsets = self._compute_offsets()

        # Initialisation du DataFrame avec les paires de référence
        reference_pairs = self._generate_reference_pairs(offsets)
        all_pairs_df = pl.DataFrame(reference_pairs, schema=["individual", "position"])

        # Ajout des paires par individu
        for individual in tqdm(
            self.metadata.individuals, desc="Processing pairs...", unit="individual"
        ):
            individual_pairs_df = self._generate_individual_pairs(individual, offsets)
            all_pairs_df = pl.concat(
                [all_pairs_df, individual_pairs_df], how="vertical_relaxed"
            )

        # Sauvegarde au format liste de tuples
        self._save_pairs(all_pairs_df)
        return all_pairs_df.rows()

    def _compute_offsets(self) -> list[int]:
        step = int(round(self.sequence_length * self.overlaping_ratio))
        return list(range(0, self.sequence_length, step))

    def _generate_reference_pairs(self, offsets: list[int]) -> dict[str, list]:
        pairs = []
        for offset in offsets:
            stop = self.reference_genome.shape[0] - offset
            stop -= stop % self.sequence_length
            real_seq_ids = range(offset, stop, self.sequence_length)
            pairs.extend([(REFERENCE_STR, i) for i in real_seq_ids])
        return {"individual": [p[0] for p in pairs], "position": [p[1] for p in pairs]}

    def _generate_individual_pairs(
        self, individual: str, offsets: list[int]
    ) -> pl.DataFrame:
        reference = self.reference_genome
        indiv_df = self.snp_per_individual[individual]
        joined = reference.join(
            indiv_df[["allele", "position"]], on="position", how="left"
        )

        if "__index_level_0__" in joined.columns:
            joined = joined.drop("__index_level_0__")

        all_rows = []

        for offset in offsets:
            sub_df = joined.slice(offset, None)

            sub_df = sub_df.with_columns(
                pl.arange(0, sub_df.height)
                .floordiv(self.sequence_length)
                .alias("seq_id")
            )

            grouped = (
                sub_df.group_by("seq_id")
                .agg(
                    [
                        pl.col("allele").drop_nulls().first().alias("allele"),
                        pl.count().alias("n_rows"),
                    ]
                )
                .filter(
                    (pl.col("allele").is_not_null())
                    & (pl.col("n_rows") == self.sequence_length)
                )
                .with_columns(
                    (pl.col("seq_id") * self.sequence_length + offset).alias(
                        "real_seq_id"
                    )
                )
                .drop("n_rows")
                .sort("seq_id")
            )

            all_rows.append(
                grouped.select(
                    pl.lit(individual).alias("individual"),
                    pl.col("real_seq_id").alias("position"),
                )
            )

        return pl.concat(all_rows, how="vertical_relaxed")

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
            return 0, 0  # Reference genome has no label
        return super().get_label(individual)

    def __hash__(self):
        individuals = sorted(self.metadata.individuals)
        individual_str = "_".join(individuals)
        s = f"{individual_str}|{self.sequence_length}|{self.overlaping_ratio}"
        return int(hashlib.md5(s.encode()).hexdigest(), 16)

    def __len__(self) -> int:
        return len(self._individual_pos_pairs)

    def __getitem__(self, idx):
        individual, snp_idx = self._individual_pos_pairs[idx].rows()[0]
        sub_ref_updated = self._extract_individual_subsequence(individual, snp_idx)
        return self._subsequence_to_dict(sub_ref_updated, individual)
