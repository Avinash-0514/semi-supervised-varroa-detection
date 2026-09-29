from pathlib import Path

# ---------------------------
# CONFIG (EDIT THIS)
# ---------------------------
ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL")  # <-- change if needed

# Set your actual split folders here
SPLITS = [
    {
        "name": "Train (GT)",
        "images_dir": ROOT / "Labelled" / "images" / "train",
        "labels_dir": ROOT / "Labelled" / "labels" / "train",
        "labels_expected": True,
    },
    {
        "name": "Validation (GT)",
        "images_dir": ROOT / "Labelled" / "images" / "val",
        "labels_dir": ROOT / "Labelled" / "labels" / "val",
        "labels_expected": True,
    },
    {
        "name": "Test (GT)",
        "images_dir": ROOT / "test" / "images",
        "labels_dir": ROOT / "test" / "labels",
        "labels_expected": False,  # test may have missing labels in your dataset
    },
    # Optional: include this if you want Unlabeled-Pool row too
    # {
    #     "name": "Unlabeled-Pool",
    #     "images_dir": ROOT / "Unlabelled" / "images",
    #     "labels_dir": None,
    #     "labels_expected": False,
    # },
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def read_label_lines(label_path: Path) -> list[str]:
    try:
        txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return []
    if not txt:
        return []
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]


def is_yolo_line(line: str) -> bool:
    parts = line.split()
    if len(parts) != 5:
        return False
    try:
        int(float(parts[0]))
        vals = list(map(float, parts[1:]))
    except Exception:
        return False
    return all(0.0 <= v <= 1.0 for v in vals)


def summarize_split(name: str, images_dir: Path, labels_dir: Path | None):
    imgs = list_images(images_dir)
    img_stems = {p.stem for p in imgs}
    images_count = len(imgs)

    has_label_file = 0
    missing_label = 0
    healthy = 0
    infected = 0
    total_boxes = 0
    bad_label_files = 0

    if labels_dir is None:
        return {
            "Split": name,
            "Images": images_count,
            "HasLabel": 0,
            "Healthy": 0,
            "Infected": 0,
            "TotalBoxes": 0,
            "MissingLabel": 0,
            "BadLabelFiles": 0,
        }

    # Iterate images (so "missing label" is correct)
    for stem in img_stems:
        lbl = labels_dir / f"{stem}.txt"
        if not lbl.exists():
            missing_label += 1
            continue

        has_label_file += 1
        lines = read_label_lines(lbl)

        # empty file => healthy
        if len(lines) == 0:
            healthy += 1
            continue

        # Validate YOLO lines; still count boxes if valid
        ok = all(is_yolo_line(ln) for ln in lines)
        if not ok:
            bad_label_files += 1
            # don’t trust its box count
            continue

        infected += 1
        total_boxes += len(lines)

    return {
        "Split": name,
        "Images": images_count,
        "HasLabel": has_label_file,
        "Healthy": healthy,
        "Infected": infected,
        "TotalBoxes": total_boxes,
        "MissingLabel": missing_label,
        "BadLabelFiles": bad_label_files,
    }


def print_table(rows: list[dict]):
    # Your exact columns
    headers = ["Split", "Images", "Has label file", "Healthy (0 box)", "Infected (≥1 box)", "Total Boxes"]
    print("\n=== DATASET USAGE SUMMARY (CURRENT EXPERIMENT) ===\n")
    print(f"| {' | '.join(headers)} |")
    print(f"|{'|'.join(['-' * (len(h) + 2) for h in headers])}|")

    for r in rows:
        print(
            f"| {r['Split']} | {r['Images']} | {r['HasLabel']} | {r['Healthy']} | {r['Infected']} | {r['TotalBoxes']} |"
        )

    # Extra diagnostics (optional but useful)
    print("\nNotes (diagnostics):")
    for r in rows:
        if r["MissingLabel"] > 0 or r["BadLabelFiles"] > 0:
            print(
                f"- {r['Split']}: Missing labels={r['MissingLabel']}, Bad label files={r['BadLabelFiles']}"
            )


def main():
    rows = []
    for s in SPLITS:
        rows.append(
            summarize_split(
                s["name"],
                s["images_dir"],
                s.get("labels_dir"),
            )
        )
    print_table(rows)


if __name__ == "__main__":
    main()