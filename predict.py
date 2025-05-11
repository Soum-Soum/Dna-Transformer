from loguru import logger
from tqdm import TqdmExperimentalWarning, tqdm
import typer

from adn.data.data import DatasetMode, load_datasets
from adn.models.transformers.bert import DnaBertForSequenceClassification
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
        ds, _ = load_datasets(
            path_helper=PathHelper(
                self.base_dir, custom_metadata_file=self.metadata_file
            ),
            sequence_length=self.sequence_length,
            train_eval_split=0,
            data_ratio_to_use=1,
            mode=DatasetMode.SEQUENTIAL_FIXED_LEN,
            labels_to_remove=self.labels_to_remove,
            individual_to_ignore=self.individuals_to_ignore,
            overlaping_ratio=self.overlaping_ratio,
        )

        model = DnaBertForSequenceClassification.from_pretrained(self.checkpoint_dir)

        predictor = Predictor(
            model=model,
            batch_size=self.batch_size,
            output_dir=(
                self.output_dir
                if self.output_dir
                else self.checkpoint_dir.parent.parent / "predictions"
            ),
        )

        predictor.predict_and_save(ds)


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
