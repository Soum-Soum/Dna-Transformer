from loguru import logger
import pandas as pd
import polars as pl
from tqdm import tqdm
import typer
from pathlib import Path
from adn.utils.paths_utils import PathHelper

app = typer.Typer()


@app.command()
def hapmap_to_parquet(
    hapmap_path: Path = typer.Option(..., help="Path to the hapmap file"),
    output_dir: Path = typer.Option(..., help="Output directory"),
    chunk_size: int = typer.Option(5000, help="Chunk size for processing"),
):
    path_helper = PathHelper(output_dir)

    output_path = path_helper.raw_chunks_dir
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for chunk_df in tqdm(
        pd.read_csv(
            hapmap_path,
            sep="\t",
            chunksize=chunk_size,
        )
    ):
        chunk_start = chunk_df.index[0]
        chunk_end = chunk_df.index[-1]

        logger.info(
            f"Processing chunk position: {chunk_start}-{chunk_start + chunk_size - 1}"
        )

        output_file = f"{output_path}/chunk_{chunk_start}_{chunk_end}.parquet"
        chunk_df.to_parquet(output_file)


if __name__ == "__main__":
    app()
