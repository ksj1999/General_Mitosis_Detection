import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from detection_dataset import CocoTileDetectionDataset, detection_collate


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on tiled MIDOG++ detection data.")
    parser.add_argument(
        "--train-coco",
        default="jun_model/demo1/detection_tiles/Aperio_CS2/train_tiles_coco.json",
        help="Training tile COCO JSON.",
    )
    parser.add_argument(
        "--test-coco",
        default="jun_model/demo1/detection_tiles/Aperio_CS2/test_tiles_coco.json",
        help="Test tile COCO JSON used for optional loss-only monitoring.",
    )
    parser.add_argument("--output-dir", default="jun_model/demo1/models/detection_frcnn_aperio", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size. Use 1 if GPU memory is limited.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="Max gradient norm. 0 disables clipping.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Keep 0 on Windows if unsure.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device.")
    parser.add_argument("--color-jitter", action="store_true", help="Use color jitter augmentation on training images.")
    parser.add_argument(
        "--augmentation",
        choices=["none", "mild", "strong"],
        default="none",
        help="Training color augmentation strength. --color-jitter is kept as an alias for strong.",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "reinhard_rgb"],
        default="none",
        help="Optional stain/color normalization applied to train and test images.",
    )
    parser.add_argument("--max-train-images", type=int, default=0, help="Limit train images for smoke tests. 0 uses all.")
    parser.add_argument("--max-test-images", type=int, default=0, help="Limit test images for loss monitoring. 0 uses all.")
    parser.add_argument("--eval-loss-every", type=int, default=1, help="Compute test loss every N epochs. 0 disables.")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization and data loading.")
    return parser.parse_args()


def make_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def maybe_subset(dataset, max_images):
    if max_images and max_images > 0:
        dataset.images = dataset.images[:max_images]
    return dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, num_workers, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(images, targets, device):
    images = [image.to(device, non_blocking=True) for image in images]
    targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
    return images, targets


def loss_dict_to_float(losses):
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def train_one_epoch(model, loader, optimizer, device, epoch, clip_grad_norm):
    model.train()
    running = {}
    total_batches = 0
    progress = tqdm(loader, desc=f"epoch {epoch} train", unit="batch")

    for images, targets in progress:
        images, targets = move_batch(images, targets, device)
        losses = model(images, targets)
        total_loss = sum(losses.values())

        optimizer.zero_grad()
        total_loss.backward()
        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        loss_values = loss_dict_to_float(losses)
        loss_values["total_loss"] = float(total_loss.detach().cpu())
        for key, value in loss_values.items():
            running[key] = running.get(key, 0.0) + value
        total_batches += 1

        progress.set_postfix(total_loss=f"{loss_values['total_loss']:.4f}")

    return {key: value / max(total_batches, 1) for key, value in running.items()}


@torch.no_grad()
def compute_loss(model, loader, device, epoch):
    # Torchvision detection models only return losses in train mode.
    was_training = model.training
    model.train()
    running = {}
    total_batches = 0
    progress = tqdm(loader, desc=f"epoch {epoch} test-loss", unit="batch")

    for images, targets in progress:
        images, targets = move_batch(images, targets, device)
        losses = model(images, targets)
        total_loss = sum(losses.values())

        loss_values = loss_dict_to_float(losses)
        loss_values["total_loss"] = float(total_loss.detach().cpu())
        for key, value in loss_values.items():
            running[key] = running.get(key, 0.0) + value
        total_batches += 1
        progress.set_postfix(total_loss=f"{loss_values['total_loss']:.4f}")

    if not was_training:
        model.eval()

    return {f"test_{key}": value / max(total_batches, 1) for key, value in running.items()}


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = make_device(args.device)
    print(f"device: {device}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)

    augmentation = "strong" if args.color_jitter else args.augmentation
    train_dataset = maybe_subset(
        CocoTileDetectionDataset(args.train_coco, augmentation=augmentation, normalization=args.normalization),
        args.max_train_images,
    )
    test_dataset = maybe_subset(
        CocoTileDetectionDataset(args.test_coco, color_jitter=False, normalization=args.normalization),
        args.max_test_images,
    )

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=train_generator,
    )
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"train images: {len(train_dataset)}", flush=True)
    print(f"test images: {len(test_dataset)}", flush=True)
    print(f"batch size: {args.batch_size}", flush=True)
    print(f"learning rate: {args.lr}", flush=True)
    print(f"clip grad norm: {args.clip_grad_norm}", flush=True)
    print(f"augmentation: {augmentation}", flush=True)
    print(f"normalization: {args.normalization}", flush=True)
    print(f"seed: {args.seed}", flush=True)

    model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, num_classes=2)
    model.to(device)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_test_loss = None
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, args.clip_grad_norm)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}}

        if args.eval_loss_every and epoch % args.eval_loss_every == 0:
            test_metrics = compute_loss(model, test_loader, device, epoch)
            row.update(test_metrics)

        row["seconds"] = round(time.time() - start, 2)
        history.append(row)
        append_csv(output_dir / "training_log.csv", row)
        (output_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(json.dumps(row, indent=2), flush=True)

        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch}.pt", model, optimizer, epoch, args, row)
            save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, args, row)

        test_loss = row.get("test_total_loss")
        if test_loss is not None and (best_test_loss is None or test_loss < best_test_loss):
            best_test_loss = test_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, row)

    save_checkpoint(output_dir / "final.pt", model, optimizer, args.epochs, args, history[-1] if history else {})
    print(f"done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
