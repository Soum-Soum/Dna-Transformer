from loguru import logger
from tqdm import tqdm
import typer
from adn.cli.hapmap_to_snp_per_individual import app as hapmap_to_spn_app
from adn.cli.hapmap_to_parquet import app as hapmap_to_parquet_app

app = typer.Typer(
    name="adn",
    help="ADN CLI for hapmap to SPN conversion",
)

app.add_typer(hapmap_to_spn_app)
app.add_typer(hapmap_to_parquet_app)

if __name__ == "__main__":
    logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level: ^12}</level>] <level>{message}</level>"
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
