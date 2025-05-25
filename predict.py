import traceback
from loguru import logger
from tqdm import TqdmExperimentalWarning, tqdm
from transformers import PreTrainedTokenizerFast
import typer

from adn.data.data import DatasetMode, load_datasets
from adn.data.data_collator import get_data_collator
from adn.models.transformers.bert import DnaBertForSequenceClassification
from adn.models.transformers.modern_bert import DnaModernBertForSequenceClassification
from adn.prediction import Predictor
from adn.utils.paths_utils import PathHelper
from adn.cli.common_args import ModelCommonArgs

app = typer.Typer()


@app.command()
class Predict(ModelCommonArgs):
    """
    Predict using a trained model.
    """

    overlaping_ratio: float = typer.Option(
        0.5,
        help="Overlapping ratio for sequences (0.0 to 1.0).",
    )

    def model_post_init(self, _):
        try:
            if self.model_type == "bert":
                model_class = DnaBertForSequenceClassification
            else:
                model_class = DnaModernBertForSequenceClassification
            model = model_class.from_pretrained(
                self.checkpoint_dir,
            )

            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                str(self.checkpoint_dir.parent.parent),
            )

            data_collator = get_data_collator(
                tokenizer=tokenizer,
            )

            ds, _ = load_datasets(
                path_helper=PathHelper(
                    self.base_dir, custom_metadata_file=self.metadata_file
                ),
                sequence_length=self.sequence_length,
                train_eval_split=0,
                data_ratio_to_use=1,
                mode=DatasetMode.SEQUENTIAL_FIXED_LEN,
                labels_to_remove=self.labels_to_remove,
                individuals_to_ignore=self.individuals_to_ignore,
                overlaping_ratio=self.overlaping_ratio,
            )

            predictor = Predictor(
                model=model,
                batch_size=self.batch_size,
                output_dir=(
                    self.output_dir
                    if self.output_dir
                    else self.checkpoint_dir.parent.parent / "predictions"
                ),
                data_collator=data_collator,
            )

            predictor.predict_and_save(ds)
        except Exception as e:
            logger.error(
                f"Error during training: {e}. Traceback: {traceback.format_exc()}"
            )
            raise e


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

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
