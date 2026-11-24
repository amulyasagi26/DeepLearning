"""Run a small validation search, then a controlled Fashion-MNIST experiment.

Use --stage smoke, tune, train, then evaluate. Only evaluate reads test examples.
Imports are local so these three Python files can share one Colab folder.
"""

import argparse
import csv
import json
import platform
import random
import shutil
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures in scripts, including headless Colab runs.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import (
    CLASS_NAMES, CONDITIONS, MARKER_SIZE, SHADES, MarkerDataset,
    load_test_data, load_training_data,
)
from model import SmallCNN


DEFAULT_SETTINGS = {
    "seed": 42, "batch_size": 64, "epochs": 5,
    "learning_rate": 0.001, "dropout": 0.25,
}
# Four trials: vary two settings while retaining the notebook's other choices.
LEARNING_RATES = (0.001, 0.0003)
DROPOUT_RATES = (0.25, 0.5)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_stage_directory(output, stage):
    directory = Path(output) / stage
    if directory.exists():
        raise FileExistsError(
            f"{directory} already exists. Preserve it by renaming that stage folder "
            "before retrying, or use a new --output folder for a new experiment."
        )
    directory.mkdir(parents=True)
    return directory


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_record(device):
    packages = {}
    for name in ("torch", "torchvision", "numpy", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {"python": platform.python_version(), "device": str(device), **packages}


def marker_specification():
    return {
        "marker_size": MARKER_SIZE, "marker_location": "top-left",
        "class_names": list(CLASS_NAMES), "shades": SHADES.tolist(),
        "wrong_marker_rule": "(true_label + 1) % 10",
    }


def make_loader(base, condition, settings, shuffle=False):
    # A separate generator keeps batch order independent of dropout's randomness.
    return DataLoader(
        MarkerDataset(base, condition), batch_size=settings["batch_size"],
        shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(settings["seed"]),
    )


def run_one_epoch(model, data_loader, loss_function, device, optimizer=None):
    """Notebook training loop: update weights only when an optimizer is supplied."""
    is_training = optimizer is not None
    model.train(is_training)  # Turns dropout off for validation and testing.
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_training):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss = loss_function(scores, labels)
            if is_training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def plot_history(history, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "validation"):
            ax.plot(epochs, [row[f"{split}_{metric}"] for row in history],
                    marker="o", label=split)
        ax.set(xlabel="Epoch", ylabel=metric.capitalize(), xticks=epochs)
        ax.legend()
        if metric == "accuracy":
            ax.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_model(train_data, validation_data, settings, condition, directory, device):
    """Train for the fixed epoch budget; save the final epoch, just like the notebook."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(settings["seed"])
    model = SmallCNN(dropout=settings["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    loss_function = nn.CrossEntropyLoss()
    train_loader = make_loader(train_data, condition, settings, shuffle=True)
    validation_loader = make_loader(validation_data, condition, settings)
    history = []
    start = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        train_loss, train_accuracy = run_one_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_one_epoch(
            model, validation_loader, loss_function, device
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
        })
        write_csv(directory / "history.csv", history)
        print(f"{directory.name} epoch {epoch}/{settings['epochs']} | "
              f"train loss {train_loss:.4f}, accuracy {train_accuracy:.4f} | "
              f"validation loss {validation_loss:.4f}, accuracy {validation_accuracy:.4f}",
              flush=True)
    elapsed = time.perf_counter() - start
    # CPU tensors let the saved weights load on either a Colab GPU or a laptop.
    torch.save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
               directory / "model.pt")
    record = {
        "settings": settings, "training_condition": condition,
        "training_examples": len(train_data), "validation_examples": len(validation_data),
        "checkpoint_rule": "final epoch", "elapsed_seconds": elapsed,
        "environment": environment_record(device), **history[-1],
    }
    write_json(directory / "record.json", record)
    plot_history(history, directory / "learning_curves.png", f"Training condition: {condition}")
    return record


def plot_marker_examples(base, path):
    # Use training images only, one example per class, shared across all rows.
    indices = {}
    for index in range(len(base)):
        _, label = base[index]
        indices.setdefault(int(label), index)
        if len(indices) == len(CLASS_NAMES):
            break
    fig, axes = plt.subplots(3, 10, figsize=(16, 6))
    for row, condition in enumerate(CONDITIONS):
        view = MarkerDataset(base, condition)
        for label, ax in enumerate(axes[row]):
            if label in indices:
                image, _ = view[indices[label]]
                ax.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(CLASS_NAMES[label], fontsize=9)
            if label == 0:
                ax.set_ylabel(condition)
    fig.suptitle("Same training images and true labels; only the corner marker changes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def smoke(data_root, output, device):
    directory = new_stage_directory(output, "smoke")
    train_data, validation_data = load_training_data(data_root, seed=42)
    settings = {**DEFAULT_SETTINGS, "epochs": 1}
    print("SMOKE CHECK: 128 training / 32 validation images; NOT project results.", flush=True)
    plot_marker_examples(train_data, directory / "marker_examples.png")
    for condition in ("none", "matching"):
        train_model(Subset(train_data, range(128)), Subset(validation_data, range(32)),
                    settings, condition, directory / condition, device)
    print("Smoke check complete. No test examples were evaluated.")


def tune(data_root, output, device):
    directory = new_stage_directory(output, "tuning")
    train_data, validation_data = load_training_data(data_root, seed=42)
    plot_marker_examples(train_data, directory / "marker_examples.png")
    rows = []
    for learning_rate in LEARNING_RATES:
        for dropout in DROPOUT_RATES:
            settings = {**DEFAULT_SETTINGS, "learning_rate": learning_rate, "dropout": dropout}
            trial = f"lr_{learning_rate}_dropout_{dropout}"
            record = train_model(train_data, validation_data, settings, "none",
                                 directory / trial, device)
            rows.append({
                "trial": trial, **settings,
                "validation_loss": record["validation_loss"],
                "validation_accuracy": record["validation_accuracy"],
                "elapsed_seconds": record["elapsed_seconds"],
            })
            write_csv(directory / "validation_comparison.csv", rows)
    # Predeclared choice: highest final validation accuracy, then lowest loss.
    best = min(rows, key=lambda row: (-row["validation_accuracy"], row["validation_loss"]))
    selection = {
        "trial": best["trial"],
        "settings": {key: best[key] for key in DEFAULT_SETTINGS},
        "selection_rule": "highest final clean validation accuracy; lowest loss breaks ties",
        "validation_accuracy": best["validation_accuracy"],
        "validation_loss": best["validation_loss"],
    }
    write_json(directory / "selection.json", selection)
    print("Selected using validation data only:", json.dumps(selection, indent=2))


def train(data_root, output, device):
    selection = read_json(Path(output) / "tuning" / "selection.json")
    selected = Path(output) / "tuning" / selection["trial"]
    settings = selection["settings"]
    clean_record = read_json(selected / "record.json")
    if clean_record["settings"] != settings or clean_record["training_condition"] != "none":
        raise ValueError("Selected baseline and settings do not match. Use a complete tuning run.")
    # Keep the paired training on the same device/software as the selected run.
    current_environment = environment_record(device)
    training_keys = ("python", "device", "torch", "torchvision", "numpy")
    if any(clean_record["environment"][key] != current_environment[key] for key in training_keys):
        raise ValueError("Training environment changed since tuning. Use the same runtime/device, "
                         "or start a new experiment folder and repeat tuning there.")
    directory = new_stage_directory(output, "final")
    train_data, validation_data = load_training_data(data_root, settings["seed"])
    shutil.copytree(selected, directory / "clean")
    train_model(train_data, validation_data, settings, "matching", directory / "marked", device)
    write_json(directory / "protocol.json", {
        "settings": settings, "selected_trial": selection["trial"],
        "train_size": len(train_data), "validation_size": len(validation_data),
        "preprocessing": "ToTensor only; pixel values in [0, 1]",
        **marker_specification(),
        "clean_checkpoint": "reused selected validation-trial final epoch",
        "prior_test_exposure": "reference notebook already reports a baseline test result",
        "environment": environment_record(device),
    })
    print("Both models saved. Settings are frozen; run --stage evaluate for final test results.")


def predict_dataset(model, loader, device):
    model.eval()
    total_loss = 0.0
    labels_list, predictions_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images.to(device))
            total_loss += nn.functional.cross_entropy(
                scores, labels.to(device), reduction="sum"
            ).item()
            labels_list.extend(labels.tolist())
            predictions_list.extend(scores.argmax(dim=1).cpu().tolist())
    labels, predictions = np.array(labels_list), np.array(predictions_list)
    return total_loss / len(labels), labels, predictions


def save_confusion(labels, predictions, directory, name):
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    rows = [{"true_class": CLASS_NAMES[i], **dict(zip(CLASS_NAMES, matrix[i].tolist()))}
            for i in range(10)]
    write_csv(directory / f"confusion_{name}.csv", rows)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Image count")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="Predicted clothing class",
           ylabel="True clothing class", title=name.replace("_", " "))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    for row in range(10):
        for column in range(10):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                    fontsize=7, color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(directory / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def evaluate(data_root, output, device):
    final = Path(output) / "final"
    protocol = read_json(final / "protocol.json")
    settings = protocol["settings"]
    if any(protocol[key] != value for key, value in marker_specification().items()):
        raise ValueError("Marker settings changed since training. Restore the recorded settings "
                         "before final evaluation, or start a new experiment.")
    directory = new_stage_directory(output, "evaluation")
    # This is the only experiment stage that loads or evaluates test examples.
    test_data = load_test_data(data_root)
    results, effects = [], []
    reference_labels = None
    for model_name in ("clean", "marked"):
        record = read_json(final / model_name / "record.json")
        expected_condition = "none" if model_name == "clean" else "matching"
        if record["settings"] != settings or record["training_condition"] != expected_condition:
            raise ValueError("Final checkpoints and protocol do not match.")
        model = SmallCNN(dropout=settings["dropout"]).to(device)
        model.load_state_dict(torch.load(final / model_name / "model.pt",
                                         map_location=device, weights_only=True))
        predictions_by_condition, accuracy_by_condition = {}, {}
        for condition in CONDITIONS:
            loss, labels, predictions = predict_dataset(
                model, make_loader(test_data, condition, settings), device
            )
            if reference_labels is None:
                reference_labels = labels.copy()
            if not np.array_equal(reference_labels, labels):
                raise ValueError("Test image order or true labels changed between conditions.")
            predictions_by_condition[condition] = predictions
            accuracy = float(np.mean(predictions == labels))
            accuracy_by_condition[condition] = accuracy
            marker_labels = labels if condition == "matching" else (labels + 1) % 10
            results.append({
                "model": model_name, "condition": condition, "images": len(labels),
                "loss": loss, "accuracy": accuracy,
                "prediction_change_vs_none": float(np.mean(
                    predictions != predictions_by_condition["none"])),
                "marker_agreement": (float(np.mean(predictions == marker_labels))
                                     if condition != "none" else ""),
            })
            save_confusion(labels, predictions, directory, f"{model_name}_{condition}")
            print(f"{model_name:6s} | {condition:8s} | loss {loss:.4f} | accuracy {accuracy:.4f}")
        write_csv(directory / f"predictions_{model_name}.csv", [
            {"test_index": i, "true_label": int(label), "wrong_marker_class": (int(label) + 1) % 10,
             **{condition: int(predictions_by_condition[condition][i]) for condition in CONDITIONS}}
            for i, label in enumerate(reference_labels)
        ])
        effects.append({
            "model": model_name,
            "matching_minus_none_percentage_points": 100 * (
                accuracy_by_condition["matching"] - accuracy_by_condition["none"]),
            "matching_minus_wrong_percentage_points": 100 * (
                accuracy_by_condition["matching"] - accuracy_by_condition["wrong"]),
        })
    write_csv(directory / "test_results.csv", results)
    write_csv(directory / "marker_effects.csv", effects)
    write_json(directory / "protocol.json", {**protocol, "test_examples": len(test_data),
                                             "evaluation_environment": environment_record(device)})
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CONDITIONS))
    for offset, name in ((-0.18, "clean"), (0.18, "marked")):
        values = [row["accuracy"] for row in results if row["model"] == name]
        ax.bar(x + offset, values, width=0.36, label=f"{name}-trained")
    ax.set(xticks=x, xticklabels=CONDITIONS, ylim=(0, 1), ylabel="Test accuracy",
           xlabel="Test marker condition", title="Same test images; true clothing labels unchanged")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "test_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved final results to {directory}. Interpret the observed values, including small/no effects.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "tune", "train", "evaluate"))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="results/run1")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)
    stages = {"smoke": smoke, "tune": tune, "train": train, "evaluate": evaluate}
    stages[args.stage](args.data_root, args.output, device)


if __name__ == "__main__":
    main()
