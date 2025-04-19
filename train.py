from typing import Optional
import typer
from pathlib import Path
import torch
import numpy as np
from adn.data.data import DatasetMode, data_collator, load_datasets
from adn.plots import plot_trainer_logs
from adn.models.bert import DnaBertConfig, DnaBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import evaluate

from adn.utils.paths_utils import PathHelper

app = typer.Typer()


@app.command()
def train_model(
    base_dir: Path = typer.Option(
        ..., help="Base data directory containing the dataset."
    ),
    output_dir: str = typer.Option(..., help="Directory to save model checkpoints."),
    run_name: str = typer.Option(..., help="Name of the run."),
    sequence_per_individual: int = typer.Option(
        250, help="Number of sequences per individual."
    ),
    sequence_length: int = typer.Option(128, help="Length of each sequence."),
    train_eval_split: float = typer.Option(
        0.1, help="Proportion of dataset for evaluation."
    ),
    epochs: int = typer.Option(20, help="Number of training epochs."),
    batch_size: int = typer.Option(256, help="Batch size for training and evaluation."),
    learning_rate: float = typer.Option(1e-3, help="Learning rate for training."),
    labels_to_remove: Optional[str] = typer.Option(
        None, help="Labels to remove from metadata."
    ),
    checkpoint_dir: Optional[str] = typer.Option(
        None, help="Path to a checkpoint to resume training from."
    ),
    individuals_to_ignore: Optional[str] = typer.Option(
        None, help="List of individuals to ignore during training."
    ),
):
    output_dir = Path(output_dir) / run_name
    assert not output_dir.exists(), f"Output directory {output_dir} already exists."

    train_ds, eval_ds = load_datasets(
        path_helper=PathHelper(base_dir),
        sequence_per_individual=sequence_per_individual,
        sequence_length=sequence_length,
        train_eval_split=train_eval_split,
        data_ratio_to_use=1,
        mode=DatasetMode.RANDOM_FIXED_LEN,
        labels_to_remove=labels_to_remove,
        individual_to_ignore=individuals_to_ignore,
    )

    dim = 256
    config = DnaBertConfig(
        vocab_size=train_ds.tokenizer.vocab_size,
        hidden_size=dim,
        intermediate_size=4 * dim,
        num_attention_heads=8,
        num_labels=len(train_ds.label_to_id),
        position_embedding_type="absolute",
        class_weights=train_ds.class_weights.tolist(),
        activation_shaping=True,
        activation_shaping_pruning_level=0.8,
    )

    if checkpoint_dir is None:
        model = DnaBertForSequenceClassification(
            config, class_weights=train_ds.class_weights
        )
    else:
        model = DnaBertForSequenceClassification.from_pretrained(
            checkpoint_dir,
            config=config,
            ignore_mismatched_sizes=True,
        )

    training_args = TrainingArguments(
        output_dir=output_dir / "checkpoints",
        eval_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="logs",
        save_strategy="epoch",
        logging_first_step=True,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine_with_restarts",
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        dataloader_num_workers=8,
        fp16=True,
        optim="adamw_torch_fused",
        remove_unused_columns=False,
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        predictions = torch.argmax(logits, dim=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    plot_trainer_logs(trainer.state.log_history, output_dir)


if __name__ == "__main__":
    app()
