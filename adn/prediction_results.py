from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


class OnDiskPredictionResults:

    def __init__(self, parquet_dir: Path):
        self.parquet_files_paths = list(parquet_dir.glob("*.parquet"))
        self.npy_files_paths = list(
            map(lambda x: x.with_suffix(".npy"), self.parquet_files_paths)
        )
        self.metadata = pd.read_csv(str(parquet_dir / "metadata.csv"), index_col=0)
        self.individual_to_label = self.metadata["GroupK4"].to_dict()
        self.embeddings_dim = np.load(self.npy_files_paths[0]).shape[1]

    def compute_centroids(self) -> dict[str, np.ndarray]:
        lables_set = set(self.metadata["GroupK4"].unique())
        centroids = {label: np.zeros(self.embeddings_dim) for label in lables_set}
        counts = {label: 0 for label in lables_set}

        for npy_file in tqdm(
            self.npy_files_paths, desc="Computing centroids", unit="file"
        ):
            individual = npy_file.stem
            label = self.individual_to_label[individual]
            embedding = np.load(npy_file)
            centroids[label] += np.sum(embedding, axis=0)
            counts[label] += embedding.shape[0]

        return {label: centroids[label] / counts[label] for label in centroids}
