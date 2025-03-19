from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def plot_trainer_logs(log_history, output_dir: Path):
    epochs = []
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    learning_rates = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            epochs.append(entry["epoch"])
            train_losses.append(entry["loss"])
            learning_rates.append(entry.get("learning_rate"))

        if "eval_loss" in entry:
            eval_losses.append((entry["epoch"], entry["eval_loss"]))

        if "eval_accuracy" in entry:
            eval_accuracies.append((entry["epoch"], entry["eval_accuracy"]))

    eval_epochs, eval_losses = zip(*eval_losses)
    acc_epochs, eval_accuracies = zip(*eval_accuracies)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(epochs, train_losses, label="Training Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(eval_epochs, eval_losses, label="Evaluation Loss", color="orange")
    axes[0, 1].set_title("Evaluation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")

    axes[1, 0].plot(
        acc_epochs, eval_accuracies, label="Evaluation Accuracy", color="green"
    )
    axes[1, 0].set_title("Evaluation Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")

    axes[1, 1].plot(epochs, learning_rates, label="Learning Rate", color="red")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")

    for ax in axes.flatten():
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_metrics.png")


def plot_tsne(
    res_df: pd.DataFrame,
    centroids: dict[str, np.ndarray] = {},
    output_dir: Path = None,
    perplexity=30,
    n_iter=300,
    random_state=42,
):
    required_columns = {"embeddings", "label_decoded", "GroupK9", "start_position"}
    missing_columns = required_columns - set(res_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in dataframe: {missing_columns}")

    # Extract data
    embeddings = np.stack(res_df["embeddings"].values)
    labels, label_classes = pd.factorize(res_df["label_decoded"])
    group_k9, group_k9_classes = pd.factorize(res_df["GroupK9"])
    positions = res_df["start_position"].values  # Numeric positions

    assert (
        embeddings.ndim == 2
    ), f"Expected embeddings to be a 2D array, got shape {embeddings.shape}"

    if len(centroids) != 0:
        assert len(centroids) == len(
            label_classes
        ), f"Centroids length {len(centroids)} does not match label classes length {len(label_classes)}"
        embeddings = np.concatenate([embeddings, np.stack(list(centroids.values()))])

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1,
    )
    tsne_results = tsne.fit_transform(embeddings)
    if len(centroids) != 0:
        tsne_results, centroids_tsne = (
            tsne_results[: -len(centroids)],
            tsne_results[-len(centroids) :],
        )
    else:
        centroids_tsne = None

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    plots_info = [
        ("Labels", labels, label_classes, "Accent"),
        ("Group K9", group_k9, group_k9_classes, "Set1"),
        ("Position", positions, None, "plasma"),
    ]

    for ax, (title, values, classes, cmap) in zip(axes, plots_info):
        scatter = ax.scatter(
            tsne_results[:, 0], tsne_results[:, 1], c=values, cmap=cmap, alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label=title)
        ax.set_title(f"t-SNE projection of SNP : Color Coding by {title}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        legend_handles = []

        # Add legend for categorical variables (labels and group_k9)
        if classes is not None:
            unique_values = np.unique(values)
            colormap = plt.get_cmap(cmap)
            colors = colormap(np.linspace(0, 1, len(unique_values)))
            legend_handles.extend(
                [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[i],
                        markersize=10,
                        label=classes[i],
                    )
                    for i in range(len(unique_values))
                ]
            )

        if centroids_tsne is not None:
            # Plot centroids in RED
            ax.scatter(
                centroids_tsne[:, 0],
                centroids_tsne[:, 1],
                c="#FF0000",
                edgecolors="black",
                marker="X",
                s=150,
                label="Centroids",
            )
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="X",
                    color="w",
                    markerfacecolor="#FF0000",
                    markersize=12,
                    label="Centroids",
                )
            )

        if legend_handles:
            ax.legend(handles=legend_handles, title=title, loc="upper right")

    plt.tight_layout()

    # Save or display the plot
    if output_dir:
        output_path = Path(output_dir) / "tsne_plot.png"
        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
    else:
        plt.show()

    plt.close()
