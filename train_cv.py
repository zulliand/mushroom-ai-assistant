"""Train and evaluate the transfer-learning mushroom image classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from utils import CV_MODEL_PATH, DATA_DIR, IMAGES_DIR, MODELS_DIR, ensure_directory, save_json, setup_logging

LOGGER = logging.getLogger("mushroom.cv")
METRICS_PATH = MODELS_DIR / "cv_metrics.json"
DEFAULT_DATA_DIR = DATA_DIR


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
):
    """Create train, validation, and test data loaders."""

    train_dataset, val_dataset, test_dataset, class_names, class_to_idx = build_datasets(
        data_dir=data_dir,
        image_size=image_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, class_names, class_to_idx


def build_model(num_classes: int) -> Tuple[nn.Module, bool]:
    """Create a ResNet18 transfer-learning model."""

    freeze_backbone = True
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception as exc:  # pragma: no cover - fallback path for offline environments
        LOGGER.warning("Pretrained weights unavailable, falling back to randomly initialized ResNet18: %s", exc)
        model = models.resnet18(weights=None)
        freeze_backbone = False

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith("fc"):
                parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model, freeze_backbone


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
    """Evaluate the model on the validation split."""

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += float(loss.item()) * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))

    return running_loss / max(1, total), correct / max(1, total)


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    image_size: int,
    freeze_backbone: bool,
    metrics: Dict[str, float],
) -> None:
    """Persist the trained CV model and its metadata."""

    ensure_directory(checkpoint_path.parent)
    torch.save(
        {
            "architecture": "resnet18",
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


def predict_image(image_path: str | Path, model_path: Path = CV_MODEL_PATH) -> Dict[str, object]:
    """Run inference on a single mushroom image."""

    if not Path(model_path).exists():
        raise FileNotFoundError(f"CV model not found at {model_path}. Train the model first.")

    checkpoint = torch.load(model_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", 224))
    class_names = list(checkpoint["class_names"])

    model, _ = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _, eval_transforms = build_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)

    predicted_label = class_names[int(index.item())]
    return {
        "predicted_class": predicted_label,
        "confidence": float(confidence.item()),
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
) -> Dict[str, object]:
    """Train the CV model and save the best checkpoint."""

    setup_logging()
    data_dir = resolve_image_root(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    train_loader, val_loader, test_loader, class_names, class_to_idx = build_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model, freeze_backbone = build_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=learning_rate)

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_state_dict = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
            }
        )
        LOGGER.info(
            "Epoch %s/%s - train loss %.4f - train acc %.4f - val loss %.4f - val acc %.4f",
            epoch,
            epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_state_dict = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    metrics = {
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_loss": float(best_val_loss),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "history": history,
    }
    save_checkpoint(model, checkpoint_path, class_names, class_to_idx, image_size, freeze_backbone, metrics)

    metadata = {
        "architecture": "resnet18",
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "image_size": image_size,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
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
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
    })

    LOGGER.info("Validation accuracy: %.4f", best_val_accuracy)
    LOGGER.info("Test accuracy: %.4f", test_accuracy)
    LOGGER.info("Saved CV checkpoint to %s", checkpoint_path)
    LOGGER.info("Saved CV metrics to %s", metrics_path)
    return {
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "metrics_path": str(metrics_path),
        "best_val_accuracy": float(best_val_accuracy),
        "test_accuracy": float(test_accuracy),
        "class_names": class_names,
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
