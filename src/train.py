"""Train the mixed synthetic/real relative 3D pose model."""

import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import (ANNOTATIONS_PATH, DATA_DIR, EPOCHS, REAL_IMAGE_DIR, SYNTHETIC_IMAGE_DIR,
                    TRAIN_BATCH_SIZE, WEIGHTS_DIR, PYTORCH_WEIGHTS_PATH, resolve_runtime)
from models.pose_net import BoxingPose3DNet
from utils.dataset import PoseDataset
from utils.loss import WeightedMPJPELoss, bone_length_loss


def split_by_sequence(items, validation_ratio=0.2):
    rng = random.Random(42)
    train_items = []
    val_items = []
    sources = sorted({item.source for item in items})
    for source in sources:
        source_items = [item for item in items if item.source == source]
        sequences = sorted({item.sequence_id for item in source_items})
        rng.shuffle(sequences)
        if len(sequences) >= 2:
            cutoff = max(1, int(len(sequences) * (1 - validation_ratio)))
            train_ids = set(sequences[:cutoff])
            train_items.extend(item for item in source_items
                               if item.sequence_id in train_ids)
            val_items.extend(item for item in source_items
                             if item.sequence_id not in train_ids)
        else:
            rng.shuffle(source_items)
            cutoff = max(1, int(len(source_items) * (1 - validation_ratio)))
            train_items.extend(source_items[:cutoff])
            val_items.extend(source_items[cutoff:])
    return train_items, val_items


def train() -> None:
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"annotation 파일이 없습니다: {ANNOTATIONS_PATH}")
    runtime = resolve_runtime()
    roots = [str(REAL_IMAGE_DIR), str(SYNTHETIC_IMAGE_DIR)]
    synthetic_annotations = DATA_DIR / "synthetic_annotations.json"
    annotation_paths = [str(ANNOTATIONS_PATH)]
    if synthetic_annotations.exists():
        annotation_paths.append(str(synthetic_annotations))
    dataset = PoseDataset(annotation_paths, roots)
    train_items, val_items = split_by_sequence(dataset.annotations)
    if not val_items:
        raise ValueError("validation sequence가 없습니다. 영상/sequence를 두 개 이상 준비하세요.")
    train_dataset = PoseDataset.__new__(PoseDataset)
    train_dataset.image_roots, train_dataset.annotations = dataset.image_roots, train_items
    val_dataset = PoseDataset.__new__(PoseDataset)
    val_dataset.image_roots, val_dataset.annotations = dataset.image_roots, val_items
    kwargs = runtime.loader_kwargs
    train_loader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=kwargs["num_workers"],
                              pin_memory=kwargs["pin_memory"])
    val_loader = DataLoader(val_dataset, TRAIN_BATCH_SIZE, shuffle=False,
                            num_workers=kwargs["num_workers"],
                            pin_memory=kwargs["pin_memory"])
    model = BoxingPose3DNet(pretrained=True).to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = WeightedMPJPELoss()
    best = float("inf")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, targets, visibility, sources in train_loader:
            images = images.to(runtime.device)
            targets = targets.to(runtime.device)
            visibility = visibility.to(runtime.device)
            prediction, _ = model(images)
            loss = criterion(prediction, targets, visibility, sources)
            loss = loss + 0.05 * bone_length_loss(prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, visibility, sources in val_loader:
                prediction, _ = model(images.to(runtime.device))
                val_loss += criterion(prediction, targets.to(runtime.device),
                                      visibility.to(runtime.device), sources).item()
        mean_val = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}: train={train_loss / len(train_loader):.4f}, "
              f"val={mean_val:.4f}")
        if mean_val < best:
            best = mean_val
            torch.save(model.state_dict(), PYTORCH_WEIGHTS_PATH)
    print(f"최적 가중치 저장: {PYTORCH_WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
