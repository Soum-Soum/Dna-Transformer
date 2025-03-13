from typing import Optional
import click
from pathlib import Path
from loguru import logger
import torch
import numpy as np
from adn.data import DatasetMode, data_collator, load_datasets
from adn.plots import plot_trainer_logs, plot_tsne
from adn.tokenizer import get_tokenizer
from adn.models.bert import CustomBertForSequenceClassification
from transformers import BertConfig, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate


def get_predictions(
    model, eval_dl: DataLoader, max_batches: Optional[int] = None
) -> list:
    res = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dl)):
            batch = {k: v.cuda() for k, v in batch.items()}
            labels = batch.pop("labels").cpu().detach().numpy()
            preds = model.embed(**batch)
            embedings = preds.pooler_output.cpu().detach().numpy()
            positions = batch["chromosome_positions"].cpu().detach().numpy()
            res += list(zip(embedings, positions, labels))
            if max_batches and i > max_batches:
                break
    return res


@click.command()
@click.option("--individuals-snp-dir", help="Directory containing individuals SNPs.")
@click.option("--metadata-path", help="Path to metadata file.")
@click.option("--output-dir", help="Directory to save model checkpoints.")
@click.option("--run-name", help="Name of the run.")
@click.option(
    "--sequence-per-individual", default=250, help="Number of sequences per individual."
)
@click.option("--sequence-length", default=128, help="Length of each sequence.")
@click.option(
    "--interval-length",
    type=int,
    help="Length of the interval to use for training.",
)
@click.option(
    "--train-eval-split", default=0.1, help="Proportion of dataset for evaluation."
)
@click.option("--epochs", default=20, help="Number of training epochs.")
@click.option(
    "--batch-size", default=256, help="Batch size for training and evaluation."
)
@click.option("--learning-rate", default=1e-3, help="Learning rate for training.")
@click.option(
    "--labels-to-remove", default=None, help="Labels to remove from metadata."
)
@click.option(
    "--checkpoint-dir",
    default=None,
    help="Path to a checkpoint to resume training from.",
)
@click.option(
    "--individuals-to-ignore",
    default=None,
    help="List of individuals to ignore during training.",
)
def train_model(
    individuals_snp_dir: str,
    metadata_path: str,
    output_dir: str,
    run_name: str,
    sequence_per_individual: int,
    sequence_length: int,
    train_eval_split: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    labels_to_remove: str,
    checkpoint_dir: str,
    individuals_to_ignore: str,
):
    output_dir = Path(output_dir) / run_name
    assert not output_dir.exists(), f"Output directory {output_dir} already exists."

    train_ds, eval_ds = load_datasets(
        individuals_snp_dir=Path(individuals_snp_dir),
        metadata_path=Path(metadata_path),
        sequence_per_individual=sequence_per_individual,
        sequence_length=sequence_length,
        train_eval_split=train_eval_split,
        data_ratio_to_use=1,
        mode=DatasetMode.RANDOM_FIXED_LEN,
        labels_to_remove=labels_to_remove,
        individual_to_ignore=individuals_to_ignore,
    )

    dim = 256
    config = BertConfig(
        vocab_size=train_ds.tokenizer.vocab_size,
        hidden_size=dim,
        intermediate_size=4 * dim,
        num_attention_heads=8,
        num_labels=len(train_ds.label_to_id),
        position_embedding_type="absolute",
    )

    if checkpoint_dir is None:
        model = CustomBertForSequenceClassification(
            config, class_weights=train_ds.class_weights
        )
    else:
        model = CustomBertForSequenceClassification.from_pretrained(
            checkpoint_dir,
            config=config,
            ignore_mismatched_sizes=True,
            class_weights=train_ds.class_weights,
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

    eval_data_loader = DataLoader(
        eval_ds, batch_size=32, collate_fn=data_collator, num_workers=4
    )
    predictions = get_predictions(model, eval_data_loader, max_batches=100)

    plot_tsne(res=predictions, output_dir=output_dir)


if __name__ == "__main__":
    train_model()
