import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from detection_dataset import CocoTileDetectionDataset, detection_collate
from demo1_evaluate_detection_frcnn import box_iou


COLORS = {
    "gt": (30, 180, 70),
    "baseline": (230, 60, 45),
    "comparison": (55, 115, 230),
}


def load_model(checkpoint_path, device):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def classify_predictions(pred_boxes, pred_scores, gt_boxes, threshold, iou_threshold):
    keep = pred_scores >= threshold
    pred_boxes = pred_boxes[keep].cpu()
    pred_scores = pred_scores[keep].cpu()
    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    gt_boxes = gt_boxes.cpu()

    tp_boxes = []
    fp_boxes = []
    matched_gt = set()

    if len(pred_boxes) and len(gt_boxes):
        ious = box_iou(pred_boxes, gt_boxes)
        for pred_index, box in enumerate(pred_boxes):
            best_iou, best_gt = torch.max(ious[pred_index], dim=0)
            best_gt = int(best_gt.item())
            if float(best_iou) >= iou_threshold and best_gt not in matched_gt:
                tp_boxes.append(box.tolist())
                matched_gt.add(best_gt)
            else:
                fp_boxes.append(box.tolist())
    else:
        fp_boxes = [box.tolist() for box in pred_boxes]

    fn_boxes = [box.tolist() for index, box in enumerate(gt_boxes) if index not in matched_gt]
    return {
        "tp": len(tp_boxes),
        "fp": len(fp_boxes),
        "fn": len(fn_boxes),
        "tp_boxes": tp_boxes,
        "fp_boxes": fp_boxes,
        "fn_boxes": fn_boxes,
        "pred_boxes": [box.tolist() for box in pred_boxes],
    }


def draw_boxes(image, boxes, color, width=3):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle([round(value) for value in box], outline=color, width=width)


def make_panel(image_path, gt_boxes, baseline_stats, comparison_stats, title):
    image = Image.open(image_path).convert("RGB")
    left = image.copy()
    right = image.copy()

    draw_boxes(left, gt_boxes, COLORS["gt"], width=2)
    draw_boxes(left, baseline_stats["pred_boxes"], COLORS["baseline"], width=3)

    draw_boxes(right, gt_boxes, COLORS["gt"], width=2)
    draw_boxes(right, comparison_stats["pred_boxes"], COLORS["comparison"], width=3)

    header_height = 44
    panel = Image.new("RGB", (left.width * 2, left.height + header_height), "white")
    panel.paste(left, (0, header_height))
    panel.paste(right, (left.width, header_height))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill=(0, 0, 0))
    draw.text(
        (8, 24),
        "Green=GT, red=baseline predictions, blue=comparison predictions",
        fill=(60, 60, 60),
    )
    return panel


def save_examples(records, output_dir, key, limit):
    selected = sorted(records, key=lambda row: row[key], reverse=True)[:limit]
    saved = []
    for index, row in enumerate(selected, start=1):
        panel = make_panel(
            row["image_path"],
            row["gt_boxes"],
            row["baseline"],
            row["comparison"],
            row["title"],
        )
        path = output_dir / f"{key}_{index:02d}.jpg"
        panel.save(path, quality=92)
        saved.append(str(path))
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Make side-by-side detector error panels.")
    parser.add_argument("--coco", default="jun_model/demo1/detection_tiles/Hamamatsu_S360/test_tiles_coco.json")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--comparison-checkpoint", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--comparison-name", default="comparison")
    parser.add_argument("--baseline-threshold", type=float, default=0.7)
    parser.add_argument("--comparison-threshold", type=float, default=0.7)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output-dir", default="jun_model/demo1/outputs/detection_error_panels_s360")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CocoTileDetectionDataset(args.coco)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )
    baseline = load_model(args.baseline_checkpoint, device)
    comparison = load_model(args.comparison_checkpoint, device)

    records = []
    image_offset = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="compare detections", unit="batch"):
            device_images = [image.to(device, non_blocking=True) for image in images]
            baseline_outputs = baseline(device_images)
            comparison_outputs = comparison(device_images)

            for item_index, (target, baseline_output, comparison_output) in enumerate(
                zip(targets, baseline_outputs, comparison_outputs)
            ):
                dataset_index = image_offset + item_index
                image_info = dataset.images[dataset_index]
                image_path = dataset.coco_path.parent / image_info["file_name"]
                gt_boxes = target["boxes"].cpu()

                baseline_stats = classify_predictions(
                    baseline_output["boxes"].detach().cpu(),
                    baseline_output["scores"].detach().cpu(),
                    gt_boxes,
                    args.baseline_threshold,
                    args.iou_threshold,
                )
                comparison_stats = classify_predictions(
                    comparison_output["boxes"].detach().cpu(),
                    comparison_output["scores"].detach().cpu(),
                    gt_boxes,
                    args.comparison_threshold,
                    args.iou_threshold,
                )
                title = (
                    f"{image_info['file_name']} | {args.baseline_name}: "
                    f"TP {baseline_stats['tp']} FP {baseline_stats['fp']} FN {baseline_stats['fn']} | "
                    f"{args.comparison_name}: TP {comparison_stats['tp']} "
                    f"FP {comparison_stats['fp']} FN {comparison_stats['fn']}"
                )
                records.append(
                    {
                        "image_path": image_path,
                        "gt_boxes": gt_boxes.tolist(),
                        "baseline": baseline_stats,
                        "comparison": comparison_stats,
                        "title": title,
                        "baseline_fp_extra": max(0, baseline_stats["fp"] - comparison_stats["fp"]),
                        "comparison_fp_extra": max(0, comparison_stats["fp"] - baseline_stats["fp"]),
                        "baseline_fn_extra": max(0, baseline_stats["fn"] - comparison_stats["fn"]),
                        "comparison_fn_extra": max(0, comparison_stats["fn"] - baseline_stats["fn"]),
                    }
                )
            image_offset += len(images)

    saved = {
        "baseline_more_fp": save_examples(records, output_dir, "baseline_fp_extra", args.limit),
        "comparison_more_fp": save_examples(records, output_dir, "comparison_fp_extra", args.limit),
        "baseline_more_fn": save_examples(records, output_dir, "baseline_fn_extra", args.limit),
        "comparison_more_fn": save_examples(records, output_dir, "comparison_fn_extra", args.limit),
    }
    summary = {
        "baseline_name": args.baseline_name,
        "comparison_name": args.comparison_name,
        "baseline_checkpoint": args.baseline_checkpoint,
        "comparison_checkpoint": args.comparison_checkpoint,
        "baseline_threshold": args.baseline_threshold,
        "comparison_threshold": args.comparison_threshold,
        "saved": saved,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
