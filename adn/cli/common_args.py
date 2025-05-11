from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_serializer
import typer


class ModelCommonArgs(BaseModel):
    base_dir: Path = typer.Option(help="Base data directory containing the dataset.")
    metadata_file: Optional[Path] = typer.Option(
        None, help="Path to the metadata file to use (will override the default one)."
    )
    output_dir: Path = typer.Option(
        Path("output"), help="Directory to save model checkpoints."
    )
    sequence_length: int = typer.Option(150, help="Length of each sequence.")
    batch_size: int = typer.Option(256, help="Batch size for training and evaluation.")
    labels_to_remove: Optional[str] = typer.Option(
        None, help="Labels to remove from metadata seperated by commas."
    )
    checkpoint_dir: Optional[Path] = typer.Option(
        None, help="Path to a checkpoint to resume training from."
    )
    individuals_to_ignore: Optional[Path] = typer.Option(
        None, help="List of individuals to ignore during training."
    )

    @field_serializer(
        "base_dir",
        "metadata_file",
        "output_dir",
        "checkpoint_dir",
    )
    def serialize_path(self, value: Path) -> str:
        return str(value)
