import typer
from adn.cli.hapmap_to_spn_parquet import app as hapmap_to_spn_app

app = typer.Typer(
    name="adn",
    help="ADN CLI for hapmap to SPN conversion",
)

app.add_typer(
    hapmap_to_spn_app,
)


if __name__ == "__main__":
    app()
