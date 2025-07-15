from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import pickle
import hashlib
import json

from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from tqdm import tqdm

from adn.plots import plot_2d_histogram, plot_confusion_matrix, plot_tsne
from adn.data.metadata import build_metadata


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


def process_file_group_centroids(
    npy_file_path: Path, group_map: dict[str, str]
) -> tuple[np.ndarray, int, str]:
    embedding = np.load(npy_file_path)
    individual = npy_file_path.stem
    group = group_map[individual]
    return np.sum(embedding, axis=0), embedding.shape[0], group


def add_distance_column(df, source_col, target_col, label_to_distance_col):
    """Ajoute une colonne de distance basée sur le mapping des labels"""
    condition = None
    for label, col in label_to_distance_col.items():
        if condition is None:
            condition = pl.when(pl.col(source_col) == label).then(pl.col(col))
        else:
            condition = condition.when(pl.col(source_col) == label).then(pl.col(col))

    return df.with_columns(condition.otherwise(None).alias(target_col))


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
        # Utilise la classe Metadata pour charger les métadonnées
        self.metadata = build_metadata(
            metadata_file_path=self.base_dir / "metadata.csv",
        )
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

    def _compute_centroids_by_group(
        self, group_type: str, cache_name: str
    ) -> dict[str, np.ndarray]:
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        logger.info(f"Computing {group_type} centroids...")
        if group_type == "label":
            group_map = self.metadata.metadata_df["label"].to_dict()
            groups_set = set(self.metadata.label_to_id.keys())
        elif group_type == "family":
            group_map = self.metadata.metadata_df["family"].to_dict()
            groups_set = set(self.metadata.family_to_id.keys())
        else:
            raise ValueError(f"Unknown group_type: {group_type}")

        centroids = {group: np.zeros(self.embeddings_dim) for group in groups_set}
        counts = {group: 0 for group in groups_set}

        partial_process_file = partial(
            process_file_group_centroids, group_map=group_map
        )

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            for embedding_sum, count, group in tqdm(
                executor.map(partial_process_file, self.npy_without_reference),
                total=len(self.npy_without_reference),
                desc=f"Computing {group_type} centroids",
                unit="file",
            ):
                centroids[group] += embedding_sum
                counts[group] += count

        final_centroids = {
            group: centroids[group] / counts[group]
            for group in centroids
            if counts[group] > 0
        }
        self._save_to_cache(cache_name, final_centroids)
        return final_centroids

    def compute_centroids(self) -> dict[str, np.ndarray]:
        return self._compute_centroids_by_group("label", "centroids")

    def compute_family_centroids(self) -> dict[str, np.ndarray]:
        return self._compute_centroids_by_group("family", "family_centroids")

    def _compute_distances_for_centroids(
        self, centroids: dict[str, np.ndarray], prefix: str
    ) -> pd.DataFrame:
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            partial_process_file_distances = partial(
                process_file_distances,
                centroids=centroids,
            )
            all_distances = []
            for result in tqdm(
                executor.map(partial_process_file_distances, self.npy_files_paths),
                total=len(self.npy_files_paths),
                desc=f"Computing distances ({prefix.rstrip('_')})",
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
            f"{prefix}{col}" for col in pivot_df.columns.get_level_values(1)
        ]
        return pivot_df

    def compute_distances(self) -> pd.DataFrame:
        # Try to load from cache first
        cached_distances = self._load_from_cache("distances")
        if cached_distances is not None:
            return cached_distances

        logger.info("Computing distances (labels and families)...")
        label_distances = self._compute_distances_for_centroids(
            self.compute_centroids(), "euclidean_distance_"
        )
        family_distances = self._compute_distances_for_centroids(
            self.compute_family_centroids(), "euclidean_family_distance_"
        )
        merged = pd.concat([label_distances, family_distances], axis=1)
        self._save_to_cache("distances", merged)
        return merged

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

        label_to_distance_col = {
            x: f"euclidean_distance_{x}" for x in errors_df["label_decoded"].unique()
        }
        errors_df = add_distance_column(
            errors_df, "label_decoded", "label_distance", label_to_distance_col
        )
        errors_df = add_distance_column(
            errors_df, "pred_decoded", "pred_label_distance", label_to_distance_col
        )

        # Save to cache
        self._save_to_cache("errors", errors_df)

        return errors_df

    def plot_tsne(
        self, sample_per_individual: int = 100, use_family_centroids: bool = False
    ) -> None:

        cached_data = self._load_from_cache(
            f"tsne_{sample_per_individual}_{use_family_centroids}"
        )
        if cached_data is None:
            rows = []
            for parquet_file_path in tqdm(self.parquet_files_paths):
                df = load_prediction_with_embeddings(parquet_file_path)
                rows.append(df.sample(sample_per_individual))

            concat = pl.concat(rows)

            # Extract data
            embeddings = np.stack(concat["embedding"].to_numpy())

            centroids = (
                self.compute_family_centroids()
                if use_family_centroids
                else self.compute_centroids()
            )
            
            embeddings = np.concatenate(
                [embeddings, np.stack(list(centroids.values()))]
            )

            tsne = TSNE(
                n_components=2,
                perplexity=30,
                n_iter=300,
                random_state=42,
                verbose=1,
                n_jobs=-1,
            )
            tsne_results = tsne.fit_transform(embeddings)
            tsne_results, centroids_tsne = (
                tsne_results[: -len(centroids)],
                tsne_results[-len(centroids) :],
            )
            self._save_to_cache(
                f"tsne_{sample_per_individual}_{use_family_centroids}",
                (concat, tsne_results, centroids_tsne),
            )
        else:
            concat, tsne_results, centroids_tsne = cached_data

        plot_tsne(
            res_df=concat.to_pandas(),
            tsne_results=tsne_results,
            centroids_tsne=centroids_tsne,
        )

    def plot_confusion_matrix(self):
        lazy_df = pl.scan_parquet(self.parquet_files_paths)

        preds = lazy_df.select(
            [
                pl.col("label_decoded"),
                pl.col("pred_decoded"),
                pl.col("family_decoded"),
                pl.col("pred_family_decoded"),
            ]
        ).collect()
        plot_confusion_matrix(
            y_true=preds["label_decoded"].to_pandas(),
            y_pred=preds["pred_decoded"].to_pandas(),
            normalize="true",
        )
        plot_confusion_matrix(
            y_true=preds["family_decoded"].to_pandas(),
            y_pred=preds["pred_family_decoded"].to_pandas(),
            normalize="true",
        )

        print(
            classification_report(
                y_true=preds["label_decoded"].to_pandas(),
                y_pred=preds["pred_decoded"].to_pandas(),
                zero_division=0,
            )
        )

        print(
            classification_report(
                y_true=preds["family_decoded"].to_pandas(),
                y_pred=preds["pred_family_decoded"].to_pandas(),
                zero_division=0,
            )
        )

    def plot_error_dist_2d_hist(self):
        errors = self.compute_errors()

        plot_2d_histogram(errors)
