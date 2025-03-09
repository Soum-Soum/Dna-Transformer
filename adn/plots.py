from pathlib import Path
import matplotlib.pyplot as plt


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


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def plot_tsne(res, output_dir: Path, perplexity=30, n_iter=300, random_state=42):
    """
    Performs dimensionality reduction using t-SNE and displays two plots:
    - One with color coding based on labels
    - One with color coding based on positions

    :param res: List of tuples (vector, position, label)
    :param perplexity: t-SNE hyperparameter (size of the neighborhood)
    :param n_iter: Number of iterations for t-SNE optimization
    :param random_state: Seed for reproducibility
    """
    vecs = np.stack([r[0] for r in res])
    labels = np.stack([r[2] for r in res])
    positions = np.stack([r[1] for r in res])

    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
    )
    tsne_results = tsne.fit_transform(vecs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # First plot: color coding based on labels
    scatter1 = ax1.scatter(
        tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis", alpha=0.6
    )
    plt.colorbar(scatter1, ax=ax1, label="Labels")
    ax1.set_title("t-SNE: Color Coding by Labels")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")

    # Second plot: color coding based on positions
    pos_values = positions[:, 0] if positions.ndim > 1 else positions
    scatter2 = ax2.scatter(
        tsne_results[:, 0], tsne_results[:, 1], c=pos_values, cmap="plasma", alpha=0.6
    )
    plt.colorbar(scatter2, ax=ax2, label="Position")
    ax2.set_title("t-SNE: Color Coding by Position")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(output_dir / "tsne_plot.png")
