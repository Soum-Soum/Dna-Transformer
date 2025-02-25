from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import typer

logger.configure(
    handlers=[dict(sink=lambda msg: tqdm.write(msg, end=""), colorize=True)]
)


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


def process_one_chunk(chunk: pd.DataFrame, output_path: Path, individuals: list[str]):
    output_path = output_path / f"chunk_{chunk.index[0]}_{chunk.index[-1]}.parquet"
    if output_path.exists():
        return
    chunk["main_allele"] = chunk[individuals].mode(axis=1)
    res = chunk.apply(lambda row: extract_spn(row, individuals), axis=1)
    res = pd.concat(res.tolist())
    res.to_parquet(output_path)


def main(
    metadata_path: Path,
    hapmap_path: Path,
    output_path: Path,
    limit: int = None,
    max_workers: int = 10,
):
    metadata_df = pd.read_csv(metadata_path, sep="\t")
    individuals = metadata_df["individual"]

    process_pool = ProcessPoolExecutor(max_workers=max_workers)

    chuncks_dir_path = output_path / "chunks"
    chuncks_dir_path.mkdir(exist_ok=True, parents=True)
    individuals_snps_dir_path = output_path / "individuals_snps"
    individuals_snps_dir_path.mkdir(exist_ok=True, parents=True)

    row_count = 0
    for chunk in tqdm(
        pd.read_csv(hapmap_path, sep="\t", chunksize=5000),
        desc="Chunks processing...",
        unit="chunk",
    ):
        logger.info(f"Processing chunk position: {chunk.index[0]}-{chunk.index[-1]}")
        process_pool.submit(process_one_chunk, chunk, chuncks_dir_path, individuals)
        # process_one_chunk(chunk, chuncks_dir_path, individuals)

        row_count += chunk.shape[0]
        if limit is not None and row_count >= limit:
            logger.info(f"Limit reached: {row_count} >= {limit} -> Stopping")
            break

    process_pool.shutdown(wait=True)

    parquets_files = chuncks_dir_path.glob("*.parquet")
    snp_df = pl.concat([pl.read_parquet(file) for file in tqdm(list(parquets_files))])
    snp_df = snp_df.rename(mapping={"__index_level_0__": "individual"})
    snp_df.write_parquet(output_path / "snp.parquet")

    for individual in tqdm(individuals):
        sub_df = snp_df.filter(pl.col("individual") == individual)
        sub_df = sub_df.sort("position")
        sub_df.write_parquet(individuals_snps_dir_path / f"{individual}.parquet")
        logger.info(f"Individual {individual} processed")


if __name__ == "__main__":
    typer.run(main)
