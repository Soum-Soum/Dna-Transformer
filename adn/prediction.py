from collections import defaultdict
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from adn.data import data_collator
from adn.models.bert import CustomBertForSequenceClassification


def collate_fn(features: list):
    predictions_batch = data_collator(features)
    individuals = [f["individual"] for f in features]
    intervals = [f["interval"] for f in features]
    return predictions_batch, {"individual": individuals, "interval": intervals}


class Predictor:

    def __init__(self, model_path: Path, batch_size: int = 256):
        self.model_path = model_path
        self.output_path = self.model_path.parent.parent
        self.batch_size = batch_size
        self.model = (
            CustomBertForSequenceClassification.from_pretrained(str(model_path))
            .eval()
            .cuda()
        )

    def build_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
        )

    def results_to_df(self, results: list, result_type: str) -> pd.DataFrame:
        df = pd.DataFrame(
            results, columns=[result_type, "individual", "interval", "label"]
        )
        df["start_position"] = df["interval"].apply(lambda x: int(x[0]))
        df["end_position"] = df["interval"].apply(lambda x: int(x[1]))
        df["interval_length"] = df["end_position"] - df["start_position"]
        df = df.drop(columns=["interval"])

        if result_type == "logits":
            df["pred"] = df["logits"].apply(lambda x: np.argmax(x))
            df["pred_prob"] = df.apply(lambda x: x["logits"][x["pred"]], axis=1)
            df["is_error"] = (df["pred"] != df["label"]).astype(int)
            df["error"] = df.apply(
                lambda x: min(x["is_error"], 1 - x["pred_prob"]), axis=1
            )
        return df

    def save_results(self, results: list, result_type: str) -> None:
        df = self.results_to_df(results, result_type)
        current_individual = df["individual"].iloc[0]
        if result_type == "logits":
            current_output_path = (
                self.output_path / "predictions" / f"{current_individual}.parquet"
            )
        else:
            current_output_path = (
                self.output_path / "embeddings" / f"{current_individual}.parquet"
            )

        current_output_path.parent.mkdir(exist_ok=True, parents=True)

        df.to_parquet(current_output_path)
        logger.info(
            f"Saved {result_type} for {current_individual} to {current_output_path}"
        )

    def process_one_batch(
        self, batch: dict[str, torch.Tensor], metadata: dict, mode: str
    ) -> list:
        batch = {k: v.cuda() for k, v in batch.items()}

        labels = batch["labels"].detach().cpu().numpy()

        if mode == "predict":
            output = self.model(**batch)
            results = (
                torch.nn.functional.softmax(output.logits, dim=1).detach().cpu().numpy()
            )
        elif mode == "embed":
            batch.pop("labels")
            preds = self.model.embed(**batch)
            results = preds.pooler_output.cpu().detach().numpy().tolist()
        else:
            raise ValueError("Mode must be 'predict' or 'embed'")

        current_batch_results = list(
            zip(results, metadata["individual"], metadata["interval"], labels)
        )

        return current_batch_results

    def process_and_save(self, dataset: Dataset, mode: str):
        dataloader = self.build_dataloader(dataset)

        individual_to_results = defaultdict(list)

        with torch.no_grad():
            for batch, metadata in dataloader:

                current_batch_results = self.process_one_batch(batch, metadata, mode)

                unique_individuals = np.unique(metadata["individual"])

                for individual in unique_individuals:
                    individual_to_results[individual] += list(
                        filter(lambda x: x[1] == individual, current_batch_results)
                    )

                to_delete = []
                for individual, results in individual_to_results.items():
                    if individual not in unique_individuals:
                        self.save_results(
                            results, "logits" if mode == "predict" else "embeddings"
                        )
                        to_delete.append(individual)

                for individual in to_delete:
                    del individual_to_results[individual]

    def predict_and_save(self, dataset: Dataset):
        self.process_and_save(dataset, mode="predict")

    def embed_and_save(self, dataset: Dataset):
        self.process_and_save(dataset, mode="embed")

    def embed_n_batches(self, dataset: Dataset, n_batches: int) -> pd.DataFrame:
        dataloader = self.build_dataloader(dataset)
        results = []
        with torch.no_grad():
            for i, (batch, metadata) in enumerate(dataloader):
                if i > n_batches:
                    break

                current_batch_results = self.process_one_batch(batch, metadata, "embed")
                results += current_batch_results

        return self.results_to_df(results, "embeddings")
