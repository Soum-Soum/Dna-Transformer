from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


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


def process_file_errors(parquet_file_path: Path, distances: pd.Series) -> pd.DataFrame:
    one_individual_predictions = pd.read_parquet(parquet_file_path)
    for col in distances.index:
        one_individual_predictions[col] = distances[col].tolist()
    errors = one_individual_predictions.query("is_error == 1")
    return errors


class OnDiskPredictionResults:

    def __init__(self, parquet_dir: Path, workers: int = 4):
        self.workers = workers
        self.parquet_files_paths = list(parquet_dir.glob("*.parquet"))
        self.npy_files_paths = list(
            map(lambda x: x.with_suffix(".npy"), self.parquet_files_paths)
        )
        self.metadata = pd.read_csv(str(parquet_dir / "metadata.csv"), index_col=0)
        self.individual_to_label = self.metadata["label"].to_dict()
        self.embeddings_dim = np.load(self.npy_files_paths[0]).shape[1]

    def compute_centroids(self) -> dict[str, np.ndarray]:
        lables_set = set(self.metadata["label"].unique())
        centroids = {label: np.zeros(self.embeddings_dim) for label in lables_set}
        counts = {label: 0 for label in lables_set}

        with ProcessPoolExecutor(max_workers=self.workers) as executor:

            partial_process_file_centroids = partial(
                process_file_centroids,
                individual_to_label=self.individual_to_label,
            )

            for result in tqdm(
                executor.map(partial_process_file_centroids, self.npy_files_paths),
                total=len(self.npy_files_paths),
                desc="Computing centroids",
                unit="file",
            ):
                embedding_sum, count, label = result
                centroids[label] += embedding_sum
                counts[label] += count
        return {label: centroids[label] / counts[label] for label in centroids}

    def compute_distances(self) -> pd.DataFrame:
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
        return pivot_df

    def compute_error(self) -> pd.DataFrame:

        distances_df = self.compute_distances()

        with ProcessPoolExecutor(max_workers=1) as executor:

            args = [
                (parquet_file_path, distances_df.loc[parquet_file_path.stem])
                for parquet_file_path in self.parquet_files_paths
            ]

            all_errors = []
            for result in tqdm(
                executor.map(
                    partial(process_file_errors),
                    *zip(*args),
                ),
                total=len(args),
                desc="Computing errors",
                unit="file",
            ):
                all_errors.append(result)

        errors_df = pd.concat(all_errors, ignore_index=True)

        return errors_df
