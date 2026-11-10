import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from detection_dataset import CocoTileDetectionDataset, detection_collate


def box_iou(box_a, box_b):
    if box_a.numel() == 0 or box_b.numel() == 0:
        return torch.zeros((box_a.shape[0], box_b.shape[0]), dtype=torch.float32)

    top_left = torch.max(box_a[:, None, :2], box_b[:, :2])
    bottom_right = torch.min(box_a[:, None, 2:], box_b[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (box_a[:, 2] - box_a[:, 0]).clamp(min=0) * (box_a[:, 3] - box_a[:, 1]).clamp(min=0)
    area_b = (box_b[:, 2] - box_b[:, 0]).clamp(min=0) * (box_b[:, 3] - box_b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b - inter
    return inter / union.clamp(min=1e-8)


def match_image(pred_boxes, pred_scores, gt_boxes, score_threshold, iou_threshold):
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep].cpu()
    pred_scores = pred_scores[keep].cpu()
    gt_boxes = gt_boxes.cpu()

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]

    matched_gt = set()
    tp = 0
    fp = 0

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    ious = box_iou(pred_boxes, gt_boxes)
    for pred_idx in range(len(pred_boxes)):
        unmatched_gt = [gt_idx for gt_idx in range(len(gt_boxes)) if gt_idx not in matched_gt]
        if not unmatched_gt:
            fp += 1
            continue
        unmatched_ious = ious[pred_idx, unmatched_gt]
        best_unmatched_iou, best_unmatched_index = torch.max(unmatched_ious, dim=0)
        best_gt = unmatched_gt[int(best_unmatched_index.item())]
        if float(best_unmatched_iou) >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def metrics(tp, fp, fn):
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on tiled mitosis detection data.")
    parser.add_argument("--checkpoint", default="demo1/models/detection_frcnn_aperio_b4/final.pt", help="Checkpoint path.")
    parser.add_argument("--coco", default="demo1/detection_tiles/Aperio_CS2/test_tiles_coco.json", help="Test tile COCO JSON.")
    parser.add_argument("--output-dir", default="demo1/models/detection_frcnn_aperio_b4/eval", help="Output directory.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device.")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for TP matching.")
    parser.add_argument(
        "--normalization",
        choices=["none", "reinhard_rgb"],
        default="none",
        help="Optional stain/color normalization applied at evaluation time.",
    )
    parser.add_argument(
        "--score-thresholds",
        default="0.05,0.1,0.2,0.3,0.5,0.7",
        help="Comma-separated score thresholds.",
    )
    parser.add_argument(
        "--allow-preprocessing-mismatch",
        action="store_true",
        help="Allow evaluation normalization to differ from the checkpoint training config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_thresholds = [float(value) for value in args.score_thresholds.split(",")]

    dataset = CocoTileDetectionDataset(args.coco, color_jitter=False, normalization=args.normalization)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_normalization = checkpoint.get("args", {}).get("normalization", "none")
    if checkpoint_normalization != args.normalization and not args.allow_preprocessing_mismatch:
        raise ValueError(
            "Normalization mismatch: checkpoint was trained with "
            f"'{checkpoint_normalization}' but evaluation requested '{args.normalization}'."
        )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    totals = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in score_thresholds}

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="evaluate", unit="batch"):
            images = [image.to(device, non_blocking=True) for image in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                gt_boxes = target["boxes"].detach().cpu()

                for threshold in score_thresholds:
                    tp, fp, fn = match_image(
                        pred_boxes=pred_boxes,
                        pred_scores=pred_scores,
                        gt_boxes=gt_boxes,
                        score_threshold=threshold,
                        iou_threshold=args.iou_threshold,
                    )
                    totals[threshold]["tp"] += tp
                    totals[threshold]["fp"] += fp
                    totals[threshold]["fn"] += fn

    rows = []
    for threshold in score_thresholds:
        row = {"score_threshold": threshold, "iou_threshold": args.iou_threshold}
        row.update(metrics(**totals[threshold]))
        rows.append(row)

    best = max(rows, key=lambda row: row["f1"])
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "best": best}, f, indent=2)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Evaluation complete")
    for row in rows:
        print(
            f"score={row['score_threshold']:.2f} "
            f"precision={row['precision']:.3f} recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} tp={row['tp']} fp={row['fp']} fn={row['fn']}"
        )
    print(f"best: {best}")
    print(f"metrics: {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
