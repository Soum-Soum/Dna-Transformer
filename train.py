import json
from pathlib import Path
import traceback
from typing import Optional
from loguru import logger
import typer
from pydantic import field_serializer
from adn.data.data import DatasetMode, load_datasets
from adn.data.data_collator import get_data_collator
from adn.data.datasets.base import DNADataset
from adn.eval.metrics import MetricCalculator
from adn.models.tokenizer import get_tokenizer
from adn.models.transformers.bert import DnaBertConfig, DnaBertForSequenceClassification
from adn.models.transformers.modern_bert import (
    DnaModernBertConfig,
    DnaModernBertForSequenceClassification,
)
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerFast

from adn.utils.paths_utils import PathHelper
from adn.cli.common_args import CommonArgs

app = typer.Typer()


@app.command()
class Train(CommonArgs):
    """
    Launch a training run for the model.
    """

    run_name: str = typer.Option(help="Name of the run.")
    sequence_per_individual: int = typer.Option(
        300, help="Number of sequences per individual."
    )
    train_eval_split: float = typer.Option(
        0.1, help="Proportion of dataset for evaluation."
    )
    epochs: int = typer.Option(20, help="Number of training epochs.")
    learning_rate: float = typer.Option(5e-5, help="Learning rate for training.")
    model_dim: int = typer.Option(
        128, help="Dimensionality of the model (hidden size)."
    )
    model_type: str = typer.Option(
        "modern_bert", help="Type de modèle à utiliser ('bert' ou 'modern_bert')."
    )
    tokenizer_path: Optional[Path] = typer.Option(
        None, help="Path to the tokenizer file."
    )
    activation_shaping_pruning_level: float = typer.Option(
        0.0,
        help="Pruning level for activation shaping (0.0 to 1.0).",
    )

    @field_serializer(
        "tokenizer_path",
    )
    def serialize_path(self, value: Path) -> str:
        return str(value)

    def build_config_and_model(
        self, train_ds: DNADataset, tokenizer: PreTrainedTokenizerFast
    ):
        common_config_args = {
            "hidden_size": self.model_dim,
            "num_attention_heads": 8,
            "activation_shaping": True,
            "activation_shaping_pruning_level": self.activation_shaping_pruning_level,
        }

        if self.model_type == "bert":

            bert_config_args = {
                "intermediate_size": self.model_dim * 4,
                "position_embedding_type": "absolute",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
            }

            total_config_args = {**common_config_args, **bert_config_args}

            config = DnaBertConfig.build(
                ds=train_ds,
                tokenizer=tokenizer,
                **total_config_args,
            )
            model_class = DnaBertForSequenceClassification
        else:
            modern_bert_config_args = {
                "intermediate_size": int(1.5 * self.model_dim),
            }
            total_config_args = {**common_config_args, **modern_bert_config_args}

            config = DnaModernBertConfig.build(
                ds=train_ds,
                tokenizer=tokenizer,
                **total_config_args,
            )

            model_class = DnaModernBertForSequenceClassification

        model = model_class(config)
        return config, model

    def get_config_and_model(
        self, train_ds: DNADataset, tokenizer: PreTrainedTokenizerFast
    ):
        if self.checkpoint_dir:
            logger.info(f"Loading model from checkpoint: {self.checkpoint_dir}")
            return self.load_config_and_model(
                activation_shaping_pruning_level=self.activation_shaping_pruning_level,
            )
        else:
            logger.info(
                "No checkpoint provided, creating a new model and training from scratch."
            )
            return self.build_config_and_model(train_ds, tokenizer)

    def model_post_init(self, _):
        try:
            output_dir = Path(self.output_dir) / self.run_name
            # assert not output_dir.exists(), f"Output directory {output_dir} already exists."
            output_dir.mkdir(parents=True, exist_ok=True)
            # with open(output_dir / "args_config.json", "w") as f:
            #    json.dump(self.model_dump(), f, indent=4)

            tokenizer = get_tokenizer(self.tokenizer_path)
            tokenizer.save_pretrained(output_dir)

            train_ds, eval_ds = load_datasets(
                path_helper=PathHelper(
                    self.base_dir, custom_metadata_file=self.metadata_file
                ),
                sequence_per_individual=self.sequence_per_individual,
                sequence_length=self.sequence_length,
                train_eval_split=self.train_eval_split,
                data_ratio_to_use=1,
                mode=DatasetMode.RANDOM_FIXED_LEN,
                labels_to_remove=self.labels_to_remove,
                individuals_to_ignore=self.individuals_to_ignore,
            )

            config, model = self.get_config_and_model(train_ds, tokenizer)
            config.save_pretrained(output_dir)

            training_args = TrainingArguments(
                output_dir=output_dir / "checkpoints",
                eval_strategy="epoch",
                save_strategy="best",
                metric_for_best_model="final_accuracy",
                save_total_limit=1,
                logging_strategy="steps",
                logging_dir=str(output_dir / "logs"),
                logging_steps=250,
                logging_first_step=True,
                num_train_epochs=self.epochs,
                lr_scheduler_type="cosine_with_restarts",
                per_device_eval_batch_size=self.batch_size,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                warmup_steps=1000,
                dataloader_num_workers=8,
                fp16=True,
                optim="adamw_torch_fused",
                remove_unused_columns=False,
                report_to=["tensorboard"],
            )

            metrics_calculator = MetricCalculator(metadata=train_ds.metadata)

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=get_data_collator(tokenizer),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                compute_metrics=metrics_calculator.compute_metrics,
            )

            trainer.train()

        except Exception as e:
            logger.error(
                f"Error during training: {e}. Traceback: {traceback.format_exc()}"
            )
            raise e


if __name__ == "__main__":
    app()
