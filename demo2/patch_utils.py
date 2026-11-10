from PIL import Image


OUTPUT_SIZE = 224


def extract_candidate_patch(image, row, scale, output_size=OUTPUT_SIZE):
    left = int(row[f"crop_left_{scale}"])
    top = int(row[f"crop_top_{scale}"])
    right = int(row[f"crop_right_{scale}"])
    bottom = int(row[f"crop_bottom_{scale}"])
    crop_width = right - left
    crop_height = bottom - top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"Invalid crop for candidate {row['candidate_id']}")

    patch = Image.new("RGB", (crop_width, crop_height), "white")
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
        source = image.crop(source_box).convert("RGB")
        patch.paste(source, (source_box[0] - left, source_box[1] - top))
    return patch.resize((output_size, output_size), Image.Resampling.LANCZOS)


def annotation_box_in_patch(row, scale, output_size=OUTPUT_SIZE):
    left = int(row[f"crop_left_{scale}"])
    top = int(row[f"crop_top_{scale}"])
    crop_width = int(row[f"crop_right_{scale}"]) - left
    crop_height = int(row[f"crop_bottom_{scale}"]) - top
    return (
        (float(row["bbox_x1"]) - left) * output_size / crop_width,
        (float(row["bbox_y1"]) - top) * output_size / crop_height,
        (float(row["bbox_x2"]) - left) * output_size / crop_width,
        (float(row["bbox_y2"]) - top) * output_size / crop_height,
    )
