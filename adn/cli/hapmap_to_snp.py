from concurrent.futures import ProcessPoolExecutor
import functools
import math
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import typer

from adn.utils.paths_utils import PathHelper


def setup_output_dirs(output_path: Path) -> tuple[Path, Path, Path]:
    chunks_output_dir = output_path / "chunks"
    main_alleles_output_dir = output_path / "main_alleles"
    snps_output_dir = output_path / "SNPs"
    chunks_output_dir.mkdir(exist_ok=True, parents=True)
    main_alleles_output_dir.mkdir(exist_ok=True, parents=True)
    snps_output_dir.mkdir(exist_ok=True, parents=True)
    return chunks_output_dir, main_alleles_output_dir, snps_output_dir


def extract_spn(row: pd.Series, individuals: list[str]):
    position = row["pos"]
    main_allele = row["main_allele"]
    row = row[individuals]
    diff = row[(row != main_allele) & (row != "NN")]
    diff = pd.DataFrame(diff)
    diff.columns = ["allele"]
    diff["main_allele"] = main_allele
    diff["position"] = position
    return diff


def filter_on_missing_data(
    chunk: pd.DataFrame,
    individuals: list[str],
    max_missing_data_percent: float,
) -> pd.DataFrame:
    before_len = len(chunk)
    missing_data_count = chunk[individuals].isna().sum(axis=1)
    missing_data_count += (chunk[individuals] == "NN").sum(axis=1)
    threshold = math.ceil(
        (max_missing_data_percent / 100) * len(individuals)
    )  # Calculate the threshold
    chunk = chunk[missing_data_count <= threshold]

    logger.info(
        f"Filtered based on missing data: before {before_len}, after {len(chunk)}"
    )
    return chunk


def filter_on_heterozygous_rate(
    chunk: pd.DataFrame,
    individuals: list[str],
    max_heterozygous_percent: float,
) -> pd.DataFrame:
    before_len = len(chunk)
    not_heterozygous_count = (
        chunk[individuals].isin(["AA", "TT", "CC", "GG"]).sum(axis=1)
    )
    heterozygous_count = len(individuals) - not_heterozygous_count
    threshold = math.ceil(
        (max_heterozygous_percent / 100) * len(individuals)
    )  # Calculate the threshold
    chunk = chunk[heterozygous_count <= threshold]
    logger.info(
        f"Filtered based on heterozygous rate: before {before_len}, after {len(chunk)}"
    )
    return chunk


def process_one_chunk(
    chunk_path: Path,
    path_helper: PathHelper,
    individuals: list[str],
    max_missing_data_percent: float,
    max_heterozygous_percent: float,
):
    logger.info(f"Processing chunk {chunk_path}")
    chunk = pd.read_parquet(chunk_path)

    start_index = chunk.index[0]
    end_index = chunk.index[-1]
    chunk_file_name = f"chunk_{start_index}_{end_index}.parquet"
    current_chunk_output_path = path_helper.chunks_output_dir / chunk_file_name
    current_main_allele_output_path = (
        path_helper.main_alleles_output_dir
        / f"main_allele_{start_index}_{end_index}.parquet"
    )
    if current_chunk_output_path.exists():
        return

    chunk = filter_on_missing_data(
        chunk,
        individuals,
        max_missing_data_percent,
    )
    chunk = filter_on_heterozygous_rate(
        chunk,
        individuals,
        max_heterozygous_percent,
    )
    chunk["main_allele"] = chunk[individuals].mode(axis=1)
    res = chunk.apply(lambda row: extract_spn(row, individuals), axis=1)
    res = pd.concat(res.tolist())
    res.to_parquet(current_chunk_output_path)
    logger.info(f"Saving chunk {current_chunk_output_path}")

    main_allele_df = chunk[["pos", "main_allele"]]
    main_allele_df = main_allele_df.rename(columns={"pos": "position"})
    main_allele_df.to_parquet(current_main_allele_output_path)
    logger.info(f"Saving main allele {current_main_allele_output_path}")


def aggregate_chunks(
    path_helper: PathHelper,
):
    parquets_files = path_helper.list_chunks_paths
    it = tqdm(list(parquets_files), desc="Aggregating chunks", unit="chunk")
    snp_df = pl.concat([pl.read_parquet(file) for file in it])
    snp_df = snp_df.rename(mapping={"__index_level_0__": "individual"})
    snp_df.write_parquet(path_helper.all_snp_file_path)
    logger.info(f"Saving all SNPs in {path_helper.all_snp_file_path}")


def aggregate_reference_genome(
    path_helper: PathHelper,
):
    parquets_files = path_helper.list_main_alleles_paths
    it = tqdm(
        list(parquets_files),
        desc="Aggregating main alleles as reference genome",
        unit="chunk",
    )
    main_allele_df = pl.concat([pl.read_parquet(file) for file in it])
    main_allele_df = main_allele_df.sort("position")
    main_allele_df.write_parquet(path_helper.all_main_alleles_file_path)
    logger.info(
        f"Saving the reference genome in {path_helper.all_main_alleles_file_path}"
    )


def split_snps_per_individual(
    path_helper: PathHelper,
    individuals: list[str],
):
    snp_df = pl.read_parquet(path_helper.all_snp_file_path)
    for individual in tqdm(individuals):
        sub_df = snp_df.filter(pl.col("individual") == individual)
        sub_df = sub_df.sort("position")
        sub_df.write_parquet(path_helper.snps_output_dir / f"{individual}.parquet")
        logger.info(f"Individual {individual} processed")


app = typer.Typer()


@app.command(
    name="hapmap-to-snp",
    help="Convert hapmap file to SPN format",
)
def hapmap_to_snp(
    metadata_path: Path = typer.Option(..., help="Path to the metadata file"),
    base_dir: Path = typer.Option(..., help="Path to the output directory"),
    limit: int = typer.Option(
        None, help="Limit the number of chunks to process (for testing purposes)"
    ),
    max_workers: int = typer.Option(
        10, help="Number of workers for processing chunks in parallel"
    ),
    max_missing_data_percent: float = typer.Option(
        1, help="the percent of missing data allowed for one SNP"
    ),
    max_heterozygous_percent: float = typer.Option(
        5, help="the percent of heterozygous data allowed for one SNP"
    ),
):

    path_helper = PathHelper(base_dir)

    metadata_df = pd.read_csv(metadata_path, sep="\t")
    if "label" not in metadata_df.columns:
        metadata_df["label"] = metadata_df["GroupK4"]

    metadata_df.to_csv(path_helper.metadata_file_path)
    exit(0)

    individuals = metadata_df["individual"]

    process_pool = ProcessPoolExecutor(max_workers=max_workers)

    path_helper.setup_output_dirs()

    process_one_chunk_partial = functools.partial(
        process_one_chunk,
        path_helper=path_helper,
        individuals=individuals,
        max_missing_data_percent=max_missing_data_percent,
        max_heterozygous_percent=max_heterozygous_percent,
    )

    chunks_paths = path_helper.list_raw_chunks_paths
    if limit is not None:
        chunks_paths = chunks_paths[:limit]

    tqdm(
        process_pool.map(process_one_chunk_partial, chunks_paths),
        total=len(chunks_paths),
        desc="Processing chunks",
        unit="chunk",
    )

    process_pool.shutdown(wait=True)

    aggregate_chunks(path_helper)
    aggregate_reference_genome(path_helper)

    split_snps_per_individual(path_helper, individuals)


if __name__ == "__main__":
    app()
