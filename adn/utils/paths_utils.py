from pathlib import Path


class PathHelper:

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    @property
    def chunks_output_dir(self) -> Path:
        return self.base_dir / "chunks"

    @property
    def main_alleles_output_dir(self) -> Path:
        return self.base_dir / "main_alleles"

    @property
    def snps_output_dir(self) -> Path:
        return self.base_dir / "SNPs"

    @property
    def all_snp_file_path(self) -> Path:
        return self.base_dir / "all_snps.parquet"

    @property
    def all_main_alleles_file_path(self) -> Path:
        return self.base_dir / "all_main_alleles.parquet"

    @property
    def metadata_file_path(self) -> Path:
        return self.base_dir / "metadata.csv"

    @property
    def list_chunks_paths(self) -> list[Path]:
        return list(sorted(self.chunks_output_dir.glob("*.parquet")))

    @property
    def list_main_alleles_paths(self) -> list[Path]:
        return list(sorted(self.main_alleles_output_dir.glob("*.parquet")))

    @property
    def list_snps_per_individual_paths(self) -> list[Path]:
        return list(sorted(self.snps_output_dir.glob("*.parquet")))

    def setup_output_dirs(self):
        self.chunks_output_dir.mkdir(exist_ok=True, parents=True)
        self.main_alleles_output_dir.mkdir(exist_ok=True, parents=True)
        self.snps_output_dir.mkdir(exist_ok=True, parents=True)
