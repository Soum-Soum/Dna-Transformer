from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import pickle
import hashlib
import json
from typing import Generator

from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from adn.plots import plot_2d_hist, plot_confusion_matrix, plot_tsne


def process_file_centroids(
    npy_file_path: Path,
    individual_to_label: dict[str, str],
) -> np.ndarray:
    embedding = np.load(npy_file_path)
    individual = npy_file_path.stem
    label = individual_to_label[individual]
    return np.sum(embedding, axis=0), embedding.shape[0], label


def process_file_distances(
    npy_file_path: Path,
    centroids: dict[str, np.ndarray],
) -> pd.DataFrame:
    embedding = np.load(npy_file_path)
    individual = npy_file_path.stem
    results = []
    for centroid_label, centroid in centroids.items():
        distance = np.linalg.norm(embedding - centroid, axis=1)
        results.append((individual, centroid_label, distance))

    return pd.DataFrame(results, columns=["individual", "centroid_label", "distance"])


def process_file_errors(
    parquet_file_path: Path, distances: dict[str, list[float]]
) -> pl.DataFrame:
    one_individual_predictions = pl.read_parquet(parquet_file_path)

    for col, values in distances.items():
        one_individual_predictions = one_individual_predictions.with_columns(
            pl.Series(name=col, values=values)
        )

    errors = one_individual_predictions.filter(pl.col("is_error") == 1)
    return errors


def load_prediction_with_embeddings(
    parquet_files_path: Path,
) -> pl.DataFrame:
    df = pl.read_parquet(parquet_files_path)
    npy_file_path = parquet_files_path.with_suffix(".npy")
    embedding = np.load(npy_file_path)
    df = df.with_columns(
        pl.Series(
            "embedding",
            embedding,
            dtype=pl.List(pl.Float64),
        )
    )
    return df


def add_distances_to_errors(errors_df: pl.DataFrame) -> pl.DataFrame:
    distance_columns = [
        col for col in errors_df.columns if col.startswith("euclidean_distance_")
    ]

    def add_distance(df: pl.DataFrame, decode_col: str, out_col: str) -> pl.DataFrame:
        distances = (
            df.unpivot(
                index=["individual", "start_position", decode_col],
                on=distance_columns,
            )
            .with_columns(
                pl.col("variable")
                .str.replace("euclidean_distance_", "")
                .alias("dist_label")
            )
            .filter(pl.col(decode_col) == pl.col("dist_label"))
            .rename({"value": out_col})
        )

        return errors_df.join(
            distances["individual", "start_position", decode_col, out_col],
            on=["individual", "start_position", decode_col],
            how="left",
        )

    errors_df = add_distance(
        errors_df,
        decode_col="label_decoded",
        out_col="label_distance",
    )
    errors_df = add_distance(
        errors_df,
        decode_col="pred_decoded",
        out_col="pred_label_distance",
    )
    return errors_df


class OnDiskPredictionResults:

    def __init__(self, base_dir: Path, workers: int = 4):
        self.base_dir = base_dir
        self.workers = workers
        self.parquet_files_paths = list(base_dir.glob("*.parquet"))
        self.npy_files_paths = list(
            map(lambda x: x.with_suffix(".npy"), self.parquet_files_paths)
        )
        self.npy_without_reference = list(
            filter(lambda x: not x.stem.endswith("reference"), self.npy_files_paths)
        )
        self.metadata = pd.read_csv(str(base_dir / "metadata.csv"), index_col=0)
        self.individual_to_label = self.metadata["label"].to_dict()
        self.embeddings_dim = np.load(self.npy_files_paths[0]).shape[1]

        # Create cache directory
        self.cache_dir = base_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Generate a hash for the current state to detect changes
        self._cache_key = self._generate_cache_key()

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on file modifications and metadata."""
        # Collect file modification times
        file_info = {}
        for file_path in self.npy_files_paths + self.parquet_files_paths:
            if file_path.exists():
                file_info[str(file_path)] = file_path.stat().st_mtime

        # Add metadata info
        metadata_path = self.base_dir / "metadata.csv"
        if metadata_path.exists():
            file_info[str(metadata_path)] = metadata_path.stat().st_mtime

        # Create hash
        hash_input = json.dumps(file_info, sort_keys=True).encode()
        return hashlib.md5(hash_input).hexdigest()

    def _get_cache_path(self, cache_name: str) -> Path:
        """Get the cache file path for a given cache name."""
        return self.cache_dir / f"{cache_name}_{self._cache_key}.pkl"

    def _load_from_cache(self, cache_name: str):
        """Load data from cache if it exists."""
        cache_path = self._get_cache_path(cache_name)
        if cache_path.exists():
            logger.info(f"Loading {cache_name} from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, cache_name: str, data):
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_name)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def compute_centroids(self) -> dict[str, np.ndarray]:
        # Try to load from cache first
        cached_centroids = self._load_from_cache("centroids")
        if cached_centroids is not None:
            return cached_centroids

        logger.info("Computing centroids...")
        lables_set = set(self.metadata["label"].unique())
        centroids = {label: np.zeros(self.embeddings_dim) for label in lables_set}
        counts = {label: 0 for label in lables_set}

        with ProcessPoolExecutor(max_workers=self.workers) as executor:

            partial_process_file_centroids = partial(
                process_file_centroids,
                individual_to_label=self.individual_to_label,
            )

            for result in tqdm(
                executor.map(
                    partial_process_file_centroids, self.npy_without_reference
                ),
                total=len(self.npy_without_reference),
                desc="Computing centroids",
                unit="file",
            ):
                embedding_sum, count, label = result
                centroids[label] += embedding_sum
                counts[label] += count

        final_centroids = {
            label: centroids[label] / counts[label] for label in centroids
        }

        # Save to cache
        self._save_to_cache("centroids", final_centroids)

        return final_centroids

    def compute_distances(self) -> pd.DataFrame:
        # Try to load from cache first
        cached_distances = self._load_from_cache("distances")
        if cached_distances is not None:
            return cached_distances

        logger.info("Computing distances...")
        centroids = self.compute_centroids()

        with ProcessPoolExecutor(max_workers=self.workers) as executor:

            partial_process_file_distances = partial(
                process_file_distances,
                centroids=centroids,
            )

            all_distances = []
            for result in tqdm(
                executor.map(partial_process_file_distances, self.npy_files_paths),
                total=len(self.npy_files_paths),
                desc="Computing distances",
                unit="file",
            ):
                all_distances.append(result)

        distances_df = pd.concat(all_distances, ignore_index=True)
        pivot_df = distances_df.pivot_table(
            index=["individual"],
            columns=["centroid_label"],
            values=["distance"],
        )
        pivot_df.columns = [
            "euclidean_distance_" + str(col)
            for col in pivot_df.columns.get_level_values(1)
        ]

        # Save to cache
        self._save_to_cache("distances", pivot_df)

        return pivot_df

    def compute_errors(self) -> pd.DataFrame:
        # Try to load from cache first
        cached_errors = self._load_from_cache("errors")
        if cached_errors is not None:
            return cached_errors

        logger.info("Computing errors...")
        distances_df = self.compute_distances()

        all_errors = []
        for parquet_file_path in tqdm(
            self.parquet_files_paths,
            desc="Computing errors",
            unit="file",
        ):
            errors = process_file_errors(
                parquet_file_path,
                distances_df.loc[parquet_file_path.stem],
            )
            all_errors.append(errors)

        errors_df = pl.concat(all_errors)
        errors_df = add_distances_to_errors(errors_df)

        # Save to cache
        self._save_to_cache("errors", errors_df)

        return errors_df

    def plot_tsne(self, sample_per_individual: int = 100) -> pl.DataFrame:

        rows = []
        for parquet_file_path in tqdm(self.parquet_files_paths):
            df = load_prediction_with_embeddings(parquet_file_path)
            rows.append(df.sample(sample_per_individual))

        concat = pl.concat(rows)

        plot_tsne(
            res_df=concat.to_pandas(),
            centroids=self.compute_centroids(),
            output_dir=None,
            perplexity=30,
            n_iter=300,
            random_state=42,
        )

    def plot_confusion_matrix(self):
        lazy_df = pl.scan_parquet(self.parquet_files_paths)

        preds = lazy_df.select(
            [
                pl.col("label_decoded"),
                pl.col("pred_decoded"),
            ]
        ).collect()
        plot_confusion_matrix(
            y_true=preds["label_decoded"].to_pandas(),
            y_pred=preds["pred_decoded"].to_pandas(),
            normalize="true",
        )

    def plot_error_dist_2d_hist(self):
        errors = self.compute_errors()

        plot_2d_hist(errors)
