from concurrent.futures import ProcessPoolExecutor
import functools
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import typer

from adn.utils.paths_utils import PathHelper

logger.configure(
    handlers=[dict(sink=lambda msg: tqdm.write(msg, end=""), colorize=True)]
)


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


def process_one_chunk(
    chunk: pd.DataFrame,
    path_helper: PathHelper,
    individuals: list[str],
):

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
    snp_df = pl.concat([pl.read_parquet(file) for file in it])])
    snp_df = snp_df.rename(mapping={"__index_level_0__": "individual"})
    snp_df.write_parquet(path_helper.all_snp_file_path)
    logger.info(f"Saving all SNPs in {path_helper.all_snp_file_path}")


def aggregate_reference_genome(
    path_helper: PathHelper,
):
    parquets_files = path_helper.list_main_alleles_paths
    it = tqdm(list(parquets_files), desc="Aggregating main alleles as reference genome", unit="chunk")
    main_allele_df = pl.concat(
        [pl.read_parquet(file) for file in it],
    )
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
    name="hapmap-to-spn",
    help="Convert hapmap file to SPN format",
)
def hapmap_to_spn(
    metadata_path: Path = typer.Option(..., help="Path to the metadata file"),
    hapmap_path: Path = typer.Option(..., help="Path to the hapmap file"),
    output_path: Path = typer.Option(..., help="Path to the output directory"),
    limit: int = typer.Option(
        None, help="Limit the number of rows to process (for testing purposes)"
    ),
    max_workers: int = typer.Option(
        10, help="Number of workers for processing chunks in parallel"
    ),
):

    path_helper = PathHelper(output_path)

    metadata_df = pd.read_csv(metadata_path, sep="\t")
    metadata_df.to_csv(path_helper.metadata_file_path)

    individuals = metadata_df["individual"]

    process_pool = ProcessPoolExecutor(max_workers=max_workers)

    path_helper.setup_output_dirs()

    # process_one_chunk_partial = functools.partial(
    #     process_one_chunk,
    #     path_helper=path_helper,
    #     individuals=individuals,
    # )

    # row_count = 0
    # for chunk in tqdm(
    #     pd.read_csv(hapmap_path, sep="\t", chunksize=5000),
    #     desc="Chunks processing...",
    #     unit="chunk",
    # ):
    #     logger.info(f"Processing chunk position: {chunk.index[0]}-{chunk.index[-1]}")
    #     process_pool.submit(process_one_chunk_partial, chunk)
    #     # process_one_chunk_partial(chunk)

    #     row_count += chunk.shape[0]
    #     if limit is not None and row_count >= limit:
    #         logger.info(f"Limit reached: {row_count} >= {limit} -> Stopping")
    #         break

    # process_pool.shutdown(wait=True)

    aggregate_chunks(path_helper)
    aggregate_reference_genome(path_helper)

    split_snps_per_individual(path_helper, individuals)


if __name__ == "__main__":
    app()
