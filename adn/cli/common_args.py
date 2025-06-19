import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_serializer
import typer

from transformers import PreTrainedTokenizerFast

from adn.models.transformers.bert import DnaBertConfig, DnaBertForSequenceClassification
from adn.models.transformers.modern_bert import (
    DnaModernBertConfig,
    DnaModernBertForSequenceClassification,
)


class CommonArgs(BaseModel):
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
    individuals_to_ignore: Optional[str] = typer.Option(
        None, help="List of individuals to ignore during training."
    )
    overlaping_ratio: float = typer.Option(
        0.5, help="Overlapping ratio for sequences (0.0 to 1.0)."
    )

    @field_serializer(
        "base_dir",
        "metadata_file",
        "output_dir",
        "checkpoint_dir",
    )
    def serialize_path(self, value: Path) -> str:
        return str(value)

    def load_config_and_model(self, **config_kwarks: dict) -> tuple:
        config_json_path = self.checkpoint_dir / "config.json"
        with open(config_json_path, "r") as f:
            config_json = json.load(f)

        model_type_to_class = {
            "bert": (DnaBertConfig, DnaBertForSequenceClassification),
            "modernbert": (
                DnaModernBertConfig,
                DnaModernBertForSequenceClassification,
            ),
        }

        

        model_type = config_json["model_type"]
        config_class, model_class = model_type_to_class[model_type]
        config = config_class.from_pretrained(
            self.checkpoint_dir,
            **config_kwarks,
        )

        model = model_class.from_pretrained(
            self.checkpoint_dir,
            config=config,
            ignore_mismatched_sizes=True,
        )

        return config, model
