from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_tsne(
    res_df: pd.DataFrame,
    centroids: dict[str, np.ndarray] = {},
    output_dir: Path = None,
    perplexity=30,
    n_iter=300,
    random_state=42,
):
    required_columns = {"embedding", "label_decoded", "start_position"}
    missing_columns = required_columns - set(res_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in dataframe: {missing_columns}")

    # Extract data
    embeddings = np.stack(res_df["embedding"].values)
    labels, label_classes = pd.factorize(res_df["label_decoded"])
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

    fig, axes = plt.subplots(1, 2, figsize=(25, 10))

    plots_info = [
        ("Labels", labels, label_classes, "Accent"),
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


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    normalize: str = None,
    output_dir: Path = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (12, 10),
):
    """
    Plot confusion matrix for predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: {'true', 'pred', 'all', None} - Normalize confusion matrix
        output_dir: Optional directory to save the plot
        title: Title for the plot
        figsize: Figure size (width, height)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    labels = sorted(y_true.unique())

    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".3f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    plt.title(f"{title}{' (Normalized)' if normalize else ''}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save or display the plot
    if output_dir:
        output_path = Path(output_dir) / "confusion_matrix.png"
        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {output_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
    else:
        plt.show()

    plt.close()


def plot_2d_hist(df: pl.DataFrame):
    df = df[["label_distance", "pred_label_distance"]].to_pandas()

    # Binning 2D des données
    heatmap, xedges, yedges = np.histogram2d(
        df["label_distance"], df["pred_label_distance"], bins=100
    )

    # Logarithme + lissage (interpolation type KDE)
    log_heatmap = np.log1p(heatmap)

    # Affichage
    plt.figure(figsize=(8, 6))
    plt.imshow(
        log_heatmap.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.colorbar(label="log(1 + density)")
    plt.xlabel("Label distance")
    plt.ylabel("Predicted label distance")
    plt.title("Heatmap densité of label vs predicted label distance")
    plt.grid(False)
    plt.show()
