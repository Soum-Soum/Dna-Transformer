from pathlib import Path
from typing import Optional
from loguru import logger
from tqdm import tqdm
import typer

from adn.data import DatasetMode, load_datasets
from adn.models.transformers.bert import DnaBertForSequenceClassification
from adn.prediction import Predictor


app = typer.Typer()


@app.command()
def predict(
    individuals_snp_dir: str = typer.Option(
        ..., help="Directory containing individuals SNPs."
    ),
    metadata_path: str = typer.Option(..., help="Path to metadata file."),
    output_dir: Optional[str] = typer.Option(
        None, help="Directory to save predictions."
    ),
    sequence_length: int = typer.Option(128, help="Length of each sequence."),
    batch_size: int = typer.Option(256, help="Batch size for training and evaluation."),
    labels_to_remove: Optional[str] = typer.Option(
        None, help="Labels to remove from metadata."
    ),
    checkpoint_dir: Optional[str] = typer.Option(
        None, help="Path to the checkpoint directory."
    ),
    individuals_to_ignore: Optional[str] = typer.Option(
        None, help="List of individuals to ignore during training."
    ),
    overlaping_ratio: float = typer.Option(
        0.5, help="Overlapping ratio between 2 consecutive sequences."
    ),
):
    ds, _ = load_datasets(
        individuals_snp_dir=Path(individuals_snp_dir),
        metadata_path=Path(metadata_path),
        sequence_length=sequence_length,
        train_eval_split=0,
        data_ratio_to_use=1,
        mode=DatasetMode.SEQUENTIAL_FIXED_LEN,
        labels_to_remove=labels_to_remove,
        individual_to_ignore=individuals_to_ignore,
        overlaping_ratio=overlaping_ratio,
    )

    model = DnaBertForSequenceClassification.from_pretrained(checkpoint_dir)

    predictor = Predictor(
        model=model,
        batch_size=batch_size,
        output_dir=(
            Path(output_dir)
            if output_dir
            else Path(checkpoint_dir).parent.parent / "predictions"
        ),
    )

    predictor.predict_and_save(ds)


if __name__ == "__main__":

    logger_format = (
        "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.configure(
        handlers=[
            dict(
                sink=lambda msg: tqdm.write(msg, end=""),
                format=logger_format,
                colorize=True,
            )
        ]
    )
    app()
