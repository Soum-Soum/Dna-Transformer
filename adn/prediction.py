from collections import defaultdict
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.rich import tqdm

from adn.data import DNADataset, data_collator
from adn.models.bert import DnaBertForSequenceClassification


def collate_fn(features: list):
    predictions_batch = data_collator(features)
    individuals = [f["individual"] for f in features]
    intervals = [f["interval"] for f in features]
    return predictions_batch, {"individual": individuals, "interval": intervals}


def compute_ennergy_score(logits: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sum(torch.exp(logits), dim=1))


class Predictor:

    def __init__(self, model: torch.nn.Module, output_dir: Path, batch_size: int = 256):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.model = model.eval().cuda()

    def _build_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
        )

    def _results_to_df(self, results: list, ds: DNADataset) -> pd.DataFrame:
        df = pd.DataFrame(
            results,
            columns=[
                "logits",
                "ennergy_scores",
                "embeddings",
                "individual",
                "interval",
                "label",
            ],
        )
        df["label_decoded"] = df["label"].apply(lambda x: ds.id_to_label[x])
        df["start_position"] = df["interval"].apply(lambda x: int(x[0]))
        df["end_position"] = df["interval"].apply(lambda x: int(x[1]))
        df["interval_length"] = df["end_position"] - df["start_position"]
        df = df.drop(columns=["interval"])

        df = df.set_index("individual").join(ds.metadata_df["GroupK9"]).reset_index()

        df["pred"] = df["logits"].apply(lambda x: np.argmax(x))
        df["pred_prob"] = df.apply(lambda x: x["logits"][x["pred"]], axis=1)
        df["is_error"] = (df["pred"] != df["label"]).astype(int)
        df["error"] = df.apply(lambda x: min(x["is_error"], 1 - x["pred_prob"]), axis=1)
        return df

    def _save_results(self, results_df: pd.DataFrame) -> None:
        current_individual = results_df["individual"].iloc[0]
        current_df_output_path = self.output_dir / f"{current_individual}.parquet"

        current_df_output_path.parent.mkdir(exist_ok=True, parents=True)
        current_embeddings_output_path = current_df_output_path.with_suffix(".npy")

        embeddings = np.stack(results_df["embeddings"].values)
        np.save(current_embeddings_output_path, embeddings)

        results_df = results_df.drop(columns=["embeddings"])
        results_df.to_parquet(current_df_output_path)
        logger.info(
            f"Saved predictions for {current_individual} to {current_df_output_path}"
        )

    def _process_one_batch(
        self, batch: dict[str, torch.Tensor], metadata: dict
    ) -> list:
        batch = {k: v.cuda() for k, v in batch.items()}

        embeddings_outputs, classifier_outputs = self.model.predict(**batch)
        logits = (
            torch.nn.functional.softmax(classifier_outputs.logits, dim=1)
            .cpu()
            .detach()
            .numpy()
        )

        ennergy_scores = (
            compute_ennergy_score(classifier_outputs.logits).cpu().detach().numpy()
        )

        embeddings = embeddings_outputs.pooler_output.cpu().detach().numpy()

        results = list(
            zip(
                logits,
                ennergy_scores,
                embeddings,
                metadata["individual"],
                metadata["interval"],
                batch["labels"].cpu().detach().numpy(),
            )
        )

        return results

    def _save_metadata(self, dataset: DNADataset) -> None:
        metadata_df = dataset.metadata_df.copy()
        metadata_df.to_csv(self.output_dir / "metadata.csv")
        logger.info(f"Saved metadata to {self.output_dir / 'metadata.csv'}")

    def predict_and_save(self, dataset: DNADataset) -> None:
        self._save_metadata(dataset)
        dataloader = self._build_dataloader(dataset)

        individual_to_results = defaultdict(list)

        with torch.no_grad():
            for batch, metadata in tqdm(
                dataloader, desc="Predicting...", unit="batch", total=len(dataloader)
            ):

                current_batch_results = self._process_one_batch(batch, metadata)

                unique_individuals = np.unique(metadata["individual"])

                for individual in unique_individuals:
                    individual_to_results[individual] += list(
                        filter(lambda x: x[3] == individual, current_batch_results)
                    )

                to_delete = []
                for individual, results in individual_to_results.items():
                    if individual not in unique_individuals:
                        results_df = self._results_to_df(results, dataset)
                        self._save_results(results_df)
                        to_delete.append(individual)

                for individual in to_delete:
                    del individual_to_results[individual]

    def predict_n_batches(self, dataset: Dataset, n_batches: int) -> pd.DataFrame:
        dataloader = self._build_dataloader(dataset)
        results = []
        with torch.no_grad():
            for i, (batch, metadata) in enumerate(tqdm(dataloader)):
                if i >= n_batches:
                    break

                current_batch_results = self._process_one_batch(batch, metadata)
                results += current_batch_results

        return self._results_to_df(results, dataset)
