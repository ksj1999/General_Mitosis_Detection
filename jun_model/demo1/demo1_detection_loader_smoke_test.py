import argparse

from torch.utils.data import DataLoader

from detection_dataset import CocoTileDetectionDataset, detection_collate


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test the tiled COCO detection dataset loader.")
    parser.add_argument(
        "--coco",
        default="jun_model/demo1/detection_tiles/Aperio_CS2/train_tiles_coco.json",
        help="Tile COCO JSON file.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = CocoTileDetectionDataset(args.coco)
    print(f"dataset images: {len(dataset)}")

    positive_index = None
    negative_index = None
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if len(target["boxes"]) > 0 and positive_index is None:
            positive_index = idx
        if len(target["boxes"]) == 0 and negative_index is None:
            negative_index = idx
        if positive_index is not None and negative_index is not None:
            break

    for name, idx in [("negative", negative_index), ("positive", positive_index)]:
        image, target = dataset[idx]
        print(f"{name} index: {idx}")
        print(f"  image shape: {tuple(image.shape)}")
        print(f"  boxes shape: {tuple(target['boxes'].shape)}")
        print(f"  labels: {target['labels'].tolist()}")
        if len(target["boxes"]) > 0:
            print(f"  first box xyxy: {target['boxes'][0].tolist()}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=detection_collate)
    images, targets = next(iter(loader))
    print(f"batch images: {len(images)}")
    print(f"batch first image shape: {tuple(images[0].shape)}")
    print(f"batch target box counts: {[len(target['boxes']) for target in targets]}")


if __name__ == "__main__":
    main()
