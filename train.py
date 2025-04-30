import json
from pathlib import Path
import traceback
from typing import Optional
from loguru import logger
import typer
from pydantic import BaseModel, field_serializer
import torch
import numpy as np
from adn.data.data import DatasetMode, get_data_collator, load_datasets
from adn.models.tokenizer import get_tokenizer
from adn.plots import plot_trainer_logs
from adn.models.base_models.bert import DnaBertConfig, DnaBertForSequenceClassification
from adn.models.base_models.modern_bert import (
    DnaModernBertConfig,
    DnaModernBertForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
import evaluate

from adn.utils.paths_utils import PathHelper

app = typer.Typer()


@app.command()
class Train(BaseModel):
    """
    Launch a training run for the model.
    """

    base_dir: Path = typer.Option(help="Base data directory containing the dataset.")
    output_dir: Path = typer.Option(
        Path("output"), help="Directory to save model checkpoints."
    )
    run_name: str = typer.Option(help="Name of the run.")

    sequence_per_individual: int = typer.Option(
        300, help="Number of sequences per individual."
    )
    sequence_length: int = typer.Option(128, help="Length of each sequence.")
    train_eval_split: float = typer.Option(
        0.1, help="Proportion of dataset for evaluation."
    )

    epochs: int = typer.Option(20, help="Number of training epochs.")
    batch_size: int = typer.Option(256, help="Batch size for training and evaluation.")
    learning_rate: float = typer.Option(1e-3, help="Learning rate for training.")
    model_dim: int = typer.Option(
        128, help="Dimensionality of the model (hidden size)."
    )

    model_type: str = typer.Option(
        "modern_bert", help="Type de modèle à utiliser ('bert' ou 'modern_bert')."
    )

    labels_to_remove: Optional[str] = typer.Option(
        None, help="Labels to remove from metadata seperated by commas."
    )
    checkpoint_dir: Optional[Path] = typer.Option(
        None, help="Path to a checkpoint to resume training from."
    )
    individuals_to_ignore: Optional[Path] = typer.Option(
        None, help="List of individuals to ignore during training."
    )
    tokenizer_path: Optional[Path] = typer.Option(
        None, help="Path to the tokenizer file."
    )
    activation_shaping_pruning_level: float = typer.Option(
        0.0,
        help="Pruning level for activation shaping (0.0 to 1.0).",
    )

    @field_serializer(
        "base_dir",
        "output_dir",
        "individuals_to_ignore",
        "checkpoint_dir",
        "tokenizer_path",
    )
    def serialize_path(self, value: Path) -> str:
        return str(value)

    def model_post_init(self, _):
        try:
            output_dir = Path(self.output_dir) / self.run_name
            # assert not output_dir.exists(), f"Output directory {output_dir} already exists."
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "config.json", "w") as f:
                json.dump(self.model_dump(), f, indent=4)

            tokenizer = get_tokenizer(self.tokenizer_path)

            train_ds, eval_ds = load_datasets(
                path_helper=PathHelper(self.base_dir),
                sequence_per_individual=self.sequence_per_individual,
                sequence_length=self.sequence_length,
                train_eval_split=self.train_eval_split,
                data_ratio_to_use=1,
                mode=DatasetMode.RANDOM_FIXED_LEN,
                labels_to_remove=self.labels_to_remove,
                individual_to_ignore=self.individuals_to_ignore,
            )

            common_config_args = {
                "vocab_size": tokenizer.vocab_size,
                "hidden_size": self.model_dim,
                "num_attention_heads": 8,
                "num_labels": len(train_ds.label_to_id),
                "class_weights": train_ds.class_weights.tolist(),
                "activation_shaping": True,
                "activation_shaping_pruning_level": self.activation_shaping_pruning_level,
                "max_position": train_ds.max_position,
            }

            if self.model_type == "bert":
                config = DnaBertConfig(
                    **common_config_args,
                    intermediate_size=self.model_dim * 4,
                    position_embedding_type="absolute",
                    hidden_dropout_prob=0,
                    attention_probs_dropout_prob=0,
                )
                the_constructor = DnaBertForSequenceClassification
            else:
                config = DnaModernBertConfig(
                    **common_config_args,
                    intermediate_size=int(1.5 * self.model_dim),
                    pad_token_id=tokenizer.pad_token_id,
                )

                the_constructor = DnaModernBertForSequenceClassification

            if self.checkpoint_dir is None:
                logger.info(
                    "No checkpoint provided, creating a new model and training from scratch."
                )
                model = the_constructor(config)
            else:
                logger.info(
                    f"Loading checkpoint from {self.checkpoint_dir} and resuming training."
                )
                model = the_constructor.from_pretrained(
                    self.checkpoint_dir,
                    config=config,
                    ignore_mismatched_sizes=True,
                )

            training_args = TrainingArguments(
                output_dir=output_dir / "checkpoints",
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="steps",
                logging_dir=str(output_dir / "logs"),
                logging_steps=1000,
                logging_first_step=True,
                num_train_epochs=self.epochs,
                lr_scheduler_type="cosine_with_restarts",
                per_device_eval_batch_size=self.batch_size,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                warmup_ratio=0.05,
                dataloader_num_workers=8,
                fp16=True,
                optim="adamw_torch_fused",
                remove_unused_columns=False,
                report_to=["tensorboard"],
            )

            accuracy_metric = evaluate.load("accuracy")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                if isinstance(logits, np.ndarray):
                    logits = torch.from_numpy(logits)
                predictions = torch.argmax(logits, dim=-1)
                return accuracy_metric.compute(
                    predictions=predictions, references=labels
                )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=get_data_collator(tokenizer),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                compute_metrics=compute_metrics,
            )

            trainer.train()

            plot_trainer_logs(trainer.state.log_history, output_dir)

        except Exception as e:
            logger.error(
                f"Error during training: {e}. Traceback: {traceback.format_exc()}"
            )
            raise e


if __name__ == "__main__":
    app()
