import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from detection_dataset import CocoTileDetectionDataset, detection_collate


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test a torchvision detector forward/backward pass.")
    parser.add_argument(
        "--coco",
        default="jun_model/demo1/detection_tiles/Aperio_CS2/train_tiles_coco.json",
        help="Tile COCO JSON file.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset = CocoTileDetectionDataset(args.coco)
    positive_index = None
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if len(target["boxes"]) > 0:
            positive_index = idx
            break
    if positive_index is None:
        raise ValueError("No positive samples found.")
    images, targets = detection_collate([dataset[positive_index]])

    model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, num_classes=2)
    model.to(device)
    model.train()

    images = [image.to(device) for image in images]
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    losses = model(images, targets)
    total_loss = sum(losses.values())
    total_loss.backward()

    print(f"device: {device}")
    for key, value in losses.items():
        print(f"{key}: {float(value.detach().cpu()):.4f}")
    print(f"total_loss: {float(total_loss.detach().cpu()):.4f}")


if __name__ == "__main__":
    main()
