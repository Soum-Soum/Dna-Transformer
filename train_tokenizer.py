from pathlib import Path
from typing import Optional
from tqdm import tqdm
import typer

from adn.data.data import DatasetMode, load_datasets
from adn.models.tokenizer import UNK_TOKEN_STR, BOS_TOKEN_STR, EOS_TOKEN_STR, CLS_TOKEN_STR
from adn.utils.paths_utils import PathHelper

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os


app = typer.Typer()


@app.command()
def train_tokenizer(
    base_dir: Path = typer.Option(
        ..., help="Base data directory containing the dataset."
    ),
    output_dir: str = typer.Option(..., help="Directory to save model checkpoints."),
    number_of_sequences: int = typer.Option(
        1000, help="Number of sequences to use for training the tokenizer."
    ),
    vocab_size: int = typer.Option(
        100, help="Maximum size of the tokenizer vocabulary."
    ),
):
    output_dir = Path(output_dir) / "tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, _ = load_datasets(
        path_helper=PathHelper(base_dir),
        sequence_per_individual=300,
        sequence_length=150,
        train_eval_split=0,
        data_ratio_to_use=1,
        mode=DatasetMode.RANDOM_FIXED_LEN,
        labels_to_remove=None,
    )

    training_file = output_dir / "tokenizer_training_data.txt"
    with open(training_file, "w") as f:
        for i in tqdm(range(number_of_sequences), desc="Writing sequences to file"):
            sequence = train_ds[i]["input_ids"]
            f.write(sequence + "\n")

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN_STR))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[UNK_TOKEN_STR, BOS_TOKEN_STR, EOS_TOKEN_STR, CLS_TOKEN_STR],
    )

    tokenizer.train([str(training_file)], trainer)

    tokenizer.save(str(output_dir / "tokenizer.json"))
    tokenizer.model.save(str(output_dir))

    typer.echo(f"Tokenizer trained and saved to {output_dir}")


if __name__ == "__main__":
    app()
