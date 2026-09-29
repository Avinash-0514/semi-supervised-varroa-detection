from pathlib import Path
from PIL import Image

# ====== CONFIG: set your ssl_dataset root here ======
#SSL_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL")  # <-- change if needed
SSL_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\New_labl")
# Extensions to search for images
IMG_EXTS = [".jpg", ".jpeg", ".png"]

def find_image_by_stem(images_dir: Path, stem: str) -> Path | None:
    """Find an image file matching the given stem in images_dir, any supported extension."""
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # fallback: sometimes stems are odd, so try glob
    matches = list(images_dir.glob(stem + ".*"))
    for m in matches:
        if m.suffix.lower() in IMG_EXTS:
            return m
    return None

def convert_one_label_file(label_path: Path, image_path: Path) -> list[str]:
    """Convert custom format (N then x1 y1 x2 y2 lines) to YOLO lines."""
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]

    if not lines:
        # empty file -> treat as no objects
        return []

    # First line is number of boxes
    try:
        n = int(float(lines[0]))
    except ValueError:
        raise ValueError(f"First line is not an integer count: {label_path.name}")

    if n == 0:
        return []

    # Load image size
    with Image.open(image_path) as im:
        w, h = im.size

    yolo_lines = []
    expected = n
    box_lines = lines[1:]

    if len(box_lines) < expected:
        raise ValueError(f"Expected {expected} box lines but found {len(box_lines)} in {label_path.name}")

    for i in range(expected):
        parts = box_lines[i].split()
        if len(parts) != 4:
            raise ValueError(f"Box line {i+1} does not have 4 values in {label_path.name}: '{box_lines[i]}'")

        x1, y1, x2, y2 = map(float, parts)

        # Convert to YOLO normalized (x_center, y_center, width, height)
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h

        # Clamp just in case of tiny floating errors
        def clamp(v): 
            return max(0.0, min(1.0, v))

        xc, yc, bw, bh = map(clamp, (xc, yc, bw, bh))

        # class_id = 0 (mite)
        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return yolo_lines

def convert_split(split_name: str, images_dir: Path, labels_dir: Path):
    print(f"\n=== Converting: {split_name} ===")
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return

    backup_dir = labels_dir.parent / f"{labels_dir.name}_original_{split_name}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    converted = 0
    skipped_no_image = 0
    failed = 0

    for label_path in labels_dir.glob("*.txt"):
        total += 1
        stem = label_path.stem
        image_path = find_image_by_stem(images_dir, stem)

        if image_path is None:
            skipped_no_image += 1
            continue

        # backup original label once
        backup_path = backup_dir / label_path.name
        if not backup_path.exists():
            backup_path.write_bytes(label_path.read_bytes())

        try:
            yolo_lines = convert_one_label_file(label_path, image_path)
            # Write YOLO label (empty file means no objects)
            label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            converted += 1
        except Exception as e:
            failed += 1
            print(f"❌ Failed: {label_path.name} -> {e}")

    print(f"Total label files: {total}")
    print(f"Converted: {converted}")
    print(f"Skipped (no matching image): {skipped_no_image}")
    print(f"Failed: {failed}")
    print(f"Backup saved in: {backup_dir}")

def main():
    # labeled train
    convert_split(
        "labeled_train",
        SSL_ROOT / "Labelled" / "images" / "train",
        SSL_ROOT / "Labelled" / "labels" / "train",
    )
    # labeled val
    convert_split(
        "labeled_val",
        SSL_ROOT / "Labelled" / "images" / "val",
        SSL_ROOT / "Labelled" / "labels" / "val",
    )
    # test
    convert_split(
        "test",
        SSL_ROOT / "test" / "images",
        SSL_ROOT / "test" / "labels",
    )

if __name__ == "__main__":
    main()
