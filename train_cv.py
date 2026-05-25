"""Train and evaluate the transfer-learning mushroom image classifier."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from utils import CV_MODEL_PATH, DATA_DIR, IMAGES_DIR, MODELS_DIR, ensure_directory, save_json, setup_logging

LOGGER = logging.getLogger("mushroom.cv")
METRICS_PATH = MODELS_DIR / "cv_metrics.json"
DEFAULT_DATA_DIR = DATA_DIR
SUPPORTED_MODEL_NAMES = {"resnet18", "efficientnet_b0"}


def resolve_image_root(data_dir: Path) -> Path:
    """Resolve the image dataset root from common project layouts."""

    candidates = [data_dir, data_dir / "images", data_dir.parent / "images"]
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "val").exists() and (candidate / "test").exists():
            return candidate
    return data_dir


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train and evaluation transforms for transfer learning."""

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, eval_transforms


def _validate_folder(folder: Path) -> None:
    """Raise a helpful error if an expected image folder is missing."""

    if not folder.exists():
        raise FileNotFoundError(f"Missing image folder: {folder}")


def build_datasets(data_dir: Path, image_size: int):
    """Build train, validation, and test datasets from folder-structured images."""

    data_dir = resolve_image_root(data_dir)
    train_transforms, eval_transforms = build_transforms(image_size)
    train_root = data_dir / "train"
    val_root = data_dir / "val"
    test_root = data_dir / "test"

    _validate_folder(train_root)
    _validate_folder(val_root)
    _validate_folder(test_root)

    train_dataset = datasets.ImageFolder(train_root, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_root, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(test_root, transform=eval_transforms)

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    if val_dataset.classes != class_names or test_dataset.classes != class_names:
        raise ValueError("Train, validation, and test folders must contain the same class subfolders.")

    return train_dataset, val_dataset, test_dataset, class_names, class_to_idx


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    random_state: int = 42,
):
    """Create train, validation, and test data loaders."""

    train_dataset, val_dataset, test_dataset, class_names, class_to_idx = build_datasets(
        data_dir=data_dir,
        image_size=image_size,
    )

    def _limit(dataset, max_samples: int | None):
        if not max_samples or max_samples <= 0 or max_samples >= len(dataset):
            return dataset, len(dataset)
        indices = list(range(len(dataset)))
        rng = random.Random(random_state)
        rng.shuffle(indices)
        selected = indices[:max_samples]
        return Subset(dataset, selected), len(selected)

    train_dataset, train_count = _limit(train_dataset, max_train_samples)
    val_dataset, val_count = _limit(val_dataset, max_val_samples)
    test_dataset, test_count = _limit(test_dataset, max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, class_names, class_to_idx, {
        "train": train_count,
        "val": val_count,
        "test": test_count,
    }


def build_model(num_classes: int, model_name: str = "resnet18") -> Tuple[nn.Module, bool]:
    """Create a transfer-learning model for the requested backbone."""

    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unsupported model_name '{model_name}'. Supported: {sorted(SUPPORTED_MODEL_NAMES)}")

    freeze_backbone = True
    model = None

    try:
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        elif model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
    except Exception as exc:  # pragma: no cover - fallback path for offline environments
        LOGGER.warning("Pretrained weights unavailable for %s, falling back to random init: %s", model_name, exc)
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
        else:
            model = models.efficientnet_b0(weights=None)
        freeze_backbone = False

    if model is None:
        raise RuntimeError(f"Could not create model for backbone: {model_name}")

    if freeze_backbone:
        if model_name == "resnet18":
            for name, parameter in model.named_parameters():
                if not name.startswith("fc"):
                    parameter.requires_grad = False
        else:
            for name, parameter in model.named_parameters():
                if not name.startswith("classifier"):
                    parameter.requires_grad = False

    if model_name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    else:
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    return model, freeze_backbone


def _topk_accuracy(outputs: torch.Tensor, labels: torch.Tensor, top_k: int) -> float:
    """Compute top-k accuracy for a batch."""

    if outputs.size(1) == 0:
        return 0.0
    k = min(top_k, outputs.size(1))
    if k <= 0:
        return 0.0
    topk_indices = torch.topk(outputs, k=k, dim=1).indices
    correct = topk_indices.eq(labels.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train a single epoch."""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.size(0))

    return running_loss / max(1, total), correct / max(1, total)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on a split and compute top-1/3/5 accuracy."""

    model.eval()
    running_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += float(loss.item()) * images.size(0)
            predictions = outputs.argmax(dim=1)
            top1_correct += int((predictions == labels).sum().item())
            top3_correct += int(
                torch.topk(outputs, k=min(3, outputs.size(1)), dim=1)
                .indices.eq(labels.unsqueeze(1))
                .any(dim=1)
                .sum()
                .item()
            )
            top5_correct += int(
                torch.topk(outputs, k=min(5, outputs.size(1)), dim=1)
                .indices.eq(labels.unsqueeze(1))
                .any(dim=1)
                .sum()
                .item()
            )
            total += int(labels.size(0))

    return (
        running_loss / max(1, total),
        top1_correct / max(1, total),
        top3_correct / max(1, total),
        top5_correct / max(1, total),
    )


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    image_size: int,
    freeze_backbone: bool,
    model_name: str,
    metrics: Dict[str, float],
) -> None:
    """Persist the trained CV model and its metadata."""

    ensure_directory(checkpoint_path.parent)
    torch.save(
        {
            "architecture": model_name,
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "image_size": image_size,
            "freeze_backbone": freeze_backbone,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def save_metrics(metrics_path: Path, payload: Dict[str, object]) -> None:
    """Persist the CV metrics JSON requested by the project spec."""

    save_json(metrics_path, payload)


def predict_image(image_path: str | Path, model_path: Path = CV_MODEL_PATH, top_k: int = 1) -> Dict[str, object]:
    """Run inference on a single mushroom image."""

    if not Path(model_path).exists():
        raise FileNotFoundError(f"CV model not found at {model_path}. Train the model first.")

    checkpoint = torch.load(model_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", 224))
    class_names = list(checkpoint["class_names"])
    model_name = str(checkpoint.get("architecture", "resnet18"))

    model, _ = build_model(num_classes=len(class_names), model_name=model_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _, eval_transforms = build_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        # top-k support
        if top_k <= 1:
            confidence, index = torch.max(probabilities, dim=0)
            predicted_label = class_names[int(index.item())]
            return {
                "predicted_class": predicted_label,
                "confidence": float(confidence.item()),
                "probabilities": probabilities.tolist(),
                "class_names": class_names,
            }
        else:
            topk = torch.topk(probabilities, k=min(top_k, probabilities.size(0)))
            values = topk.values.tolist()
            indices = topk.indices.tolist()
            top_predictions = [
                {"class": class_names[int(i)], "confidence": float(v)} for i, v in zip(indices, values)
            ]
            return {
                "top_k": top_predictions,
                "probabilities": probabilities.tolist(),
                "class_names": class_names,
            }


def train_cv_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    checkpoint_path: Path = CV_MODEL_PATH,
    metadata_path: Path | None = None,
    metrics_path: Path = METRICS_PATH,
    image_size: int = 224,
    batch_size: int = 16,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    random_state: int = 42,
    num_workers: int = 0,
    early_stopping: bool = False,
    patience: int = 3,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    model_name: str = "resnet18",
) -> Dict[str, object]:
    """Train the CV model and save the best checkpoint."""

    setup_logging()
    data_dir = resolve_image_root(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Using backbone: %s", model_name)

    # automatic batch size selection if batch_size is None or 0
    if not batch_size:
        # favor larger batches on GPU, modest on CPU
        batch_size = 64 if torch.cuda.is_available() else 16

    train_loader, val_loader, test_loader, class_names, class_to_idx, effective_sizes = build_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        random_state=random_state,
    )

    LOGGER.info(
        "Effective dataset sizes - train: %s, val: %s, test: %s",
        effective_sizes["train"],
        effective_sizes["val"],
        effective_sizes["test"],
    )


    model, freeze_backbone = build_model(num_classes=len(class_names), model_name=model_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=learning_rate)

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_state_dict = None
    history: List[Dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_top1, val_top3, val_top5 = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_loss": float(val_loss),
                "val_top1_accuracy": float(val_top1),
                "val_top3_accuracy": float(val_top3),
                "val_top5_accuracy": float(val_top5),
            }
        )
        LOGGER.info(
            "Epoch %s/%s - train loss %.4f - train acc %.4f - val loss %.4f - top1 %.4f - top3 %.4f - top5 %.4f",
            epoch,
            epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_top1,
            val_top3,
            val_top5,
        )

        # track best
        if val_top1 >= best_val_accuracy:
            best_val_accuracy = val_top1
            best_val_loss = val_loss
            best_state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # early stopping
        if early_stopping and epochs_without_improvement >= patience:
            LOGGER.info("Early stopping triggered (no improvement for %s epochs)", patience)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_top1, test_top3, test_top5 = evaluate(model, test_loader, criterion, device)
    metrics = {
        "model_name": model_name,
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_loss": float(best_val_loss),
        "top1_accuracy": float(test_top1),
        "top3_accuracy": float(test_top3),
        "top5_accuracy": float(test_top5),
        "test_accuracy": float(test_top1),
        "test_loss": float(test_loss),
        "history": history,
        "effective_dataset_sizes": effective_sizes,
    }
    save_checkpoint(model, checkpoint_path, class_names, class_to_idx, image_size, freeze_backbone, model_name, metrics)

    metadata = {
        "architecture": "resnet18",
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "image_size": image_size,
        "model_name": model_name,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_top1,
        "history": history,
    }
    if metadata_path is None:
        metadata_path = checkpoint_path.with_name("mushroom_cv_metadata.json")
    save_json(metadata_path, metadata)
    save_metrics(metrics_path, {
        "dataset_path": str(data_dir),
        "class_names": class_names,
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_loss": float(best_val_loss),
        "top1_accuracy": float(test_top1),
        "top3_accuracy": float(test_top3),
        "top5_accuracy": float(test_top5),
        "test_accuracy": float(test_top1),
        "test_loss": float(test_loss),
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "model_name": model_name,
    })

    LOGGER.info("Validation accuracy: %.4f", best_val_accuracy)
    LOGGER.info("Test top-1 accuracy: %.4f", test_top1)
    LOGGER.info("Test top-3 accuracy: %.4f", test_top3)
    LOGGER.info("Test top-5 accuracy: %.4f", test_top5)
    LOGGER.info("Saved CV checkpoint to %s", checkpoint_path)
    LOGGER.info("Saved CV metrics to %s", metrics_path)
    return {
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "metrics_path": str(metrics_path),
        "best_val_accuracy": float(best_val_accuracy),
        "top1_accuracy": float(test_top1),
        "top3_accuracy": float(test_top3),
        "top5_accuracy": float(test_top5),
        "test_accuracy": float(test_top1),
        "class_names": class_names,
        "model_name": model_name,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for CV training."""

    parser = argparse.ArgumentParser(description="Train the mushroom image classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Project data root or image root containing train/val/test")
    parser.add_argument("--checkpoint-path", type=Path, default=CV_MODEL_PATH, help="Path to save the CV checkpoint")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional path to save CV metadata")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH, help="Path to save CV metrics JSON")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for transfer learning")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    try:
        result = train_cv_pipeline(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint_path,
            metadata_path=args.metadata_path,
            metrics_path=args.metrics_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
            num_workers=args.num_workers,
        )
        LOGGER.info("Training complete: %s", result)
    except Exception as exc:  # pragma: no cover - explicit CLI error surface
        LOGGER.exception("CV training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
