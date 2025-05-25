from tqdm.rich import tqdm
from adn.data.metadata import Metadata
from adn.utils.paths_utils import PathHelper


import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import Dataset


def load_snp_per_individual(
    path_helper: PathHelper, individuals: list[str]
) -> dict[str, pl.DataFrame]:

    individuals_snp_files_paths = list(
        filter(
            lambda x: x.exists(),
            map(
                lambda x: path_helper.snp_per_individual_dir / f"{x}.parquet",
                individuals,
            ),
        ),
    )

    dataframes = {}
    for file in tqdm(individuals_snp_files_paths, desc="Loading SNP data..."):
        individual = file.stem
        dataframes[individual] = pl.read_parquet(file, use_pyarrow=True)

    return dataframes


def load_ref_genome(path_helper: PathHelper) -> pl.DataFrame:
    reference_genome = pd.read_parquet(path_helper.all_main_alleles_file_path)
    reference_genome = reference_genome.sort_values("position").reset_index(drop=True)
    return pl.from_pandas(reference_genome)


class DNADataset(Dataset):

    def __init__(
        self,
        metadata: Metadata,
        path_helper: PathHelper,
        sequence_length: int,
    ):
        super().__init__()
        self.metadata = metadata
        self.snp_per_individual = load_snp_per_individual(
            path_helper, self.metadata.individuals
        )
        self.reference_genome = load_ref_genome(path_helper)
        self.id_to_postion = self.reference_genome["position"].to_pandas().to_dict()
        self.position_to_id = {v: k for k, v in self.id_to_postion.items()}
        self.sequence_length = sequence_length
        self.max_position = self.reference_genome["position"].max()

    @property
    def snp_count(self) -> int:
        return len(self.id_to_postion)

    def get_label(self, individual: str) -> tuple[int, int]:
        label, family = self.metadata.metadata_df.loc[individual, ["label", "family"]]
        label_id = self.metadata.label_to_id[label]
        family_id = self.metadata.family_to_id[family]
        return label_id, family_id

    def _extract_individual_subsequence(
        self, individual: str, snp_idx: int
    ) -> pl.DataFrame:
        sub_ref = self.reference_genome[snp_idx : snp_idx + self.sequence_length]
        start_pos = sub_ref["position"][0]
        end_pos = sub_ref["position"][-1]

        individual_df = self.snp_per_individual[individual]
        sub_individual = individual_df.filter(
            (individual_df["position"] >= start_pos)
            & (individual_df["position"] <= end_pos)
        )
        sub_ref_updated = sub_ref.join(
            sub_individual[["allele", "position"]], on="position", how="left"
        )
        sub_ref_updated = sub_ref_updated.with_columns(
            pl.col("allele").fill_null(pl.col("main_allele"))
        )
        return sub_ref_updated

    def _subsequence_to_dict(self, sub_df: pl.DataFrame, individual: str) -> dict:
        snp_positions = sub_df["position"].to_numpy().astype(np.int32).tolist()
        snp_positions = sum(
            [
                [snp_positions[0]],
                snp_positions,
                [snp_positions[-1]],
            ],
            [],
        )

        snp_ids = [self.position_to_id[(position)] for position in snp_positions]

        sequence = (
            sub_df[["main_allele", "allele"]]
            .map_rows(lambda x: "".join(x))
            .to_numpy()
            .squeeze()
            .tolist()
        )
        sequence = " ".join(sequence)

        label_id, family_id = self.get_label(individual)

        return {
            "input_ids": sequence,
            "labels": label_id,
            "family": family_id,
            "snp_positions": snp_positions,
            "snp_ids": snp_ids,
            "interval": (snp_positions[0], snp_positions[-1]),
            "individual": individual,
        }
