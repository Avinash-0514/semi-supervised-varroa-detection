from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# =========================
# CONFIG (EDIT ONLY THIS)
# =========================
PROJECT_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo")
SSL_ROOT = PROJECT_ROOT / "SSL"

# Labeled data (ground-truth)
LABELED_TRAIN_IMG = SSL_ROOT / "Labelled" / "images" / "train"
LABELED_TRAIN_LAB = SSL_ROOT / "Labelled" / "labels" / "train"
LABELED_VAL_IMG   = SSL_ROOT / "Labelled" / "images" / "val"
LABELED_VAL_LAB   = SSL_ROOT / "Labelled" / "labels" / "val"

# Test data (if you have it in SSL/test)
TEST_IMG = SSL_ROOT / "test" / "images"
TEST_LAB = SSL_ROOT / "test" / "labels"

# Unlabeled pool (before pseudo-labeling)
UNLABELED_IMG = SSL_ROOT / "unlabelled" / "images" / "all"

# Optional: if you want to include pseudo-label dirs in report (set to None to skip)
PSEUDO_DIRS = [
    # SSL_ROOT / "pseudo_conf025",
    # SSL_ROOT / "pseudo_MT_round1",
    # SSL_ROOT / "pseudo_MT_round2",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# =========================


@dataclass
class SplitStats:
    name: str
    total_images: int = 0

    # label pairing
    images_with_label_file: int = 0
    images_missing_label_file: int = 0

    # "bee health" interpretation from labels
    infected_images: int = 0          # label file has >=1 bbox line
    healthy_images: int = 0           # label file exists but empty (0 bbox)
    unlabeled_images: int = 0         # no label file (only relevant if label dir exists but missing some)

    # bbox stats
    total_boxes: int = 0
    bad_label_files: int = 0          # label files with invalid format
    bad_label_examples: List[str] = None

    def __post_init__(self):
        if self.bad_label_examples is None:
            self.bad_label_examples = []


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def list_images(img_dir: Path) -> List[Path]:
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.is_file() and is_image(p)])


def parse_yolo_label_file(txt_path: Path) -> Tuple[bool, int]:
    """
    Returns (ok, num_boxes).
    ok=False if file contains non-YOLO lines.
    YOLO bbox line expected: class xc yc w h
    """
    try:
        raw = txt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return False, 0

    if raw == "":
        return True, 0

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    boxes = 0
    for ln in lines:
        parts = ln.split()
        # Expect 5 tokens for standard YOLO detection labels
        if len(parts) != 5:
            return False, 0
        try:
            _cls = int(float(parts[0]))
            _ = [float(x) for x in parts[1:]]
        except Exception:
            return False, 0
        boxes += 1
    return True, boxes


def analyze_split(name: str, img_dir: Path, label_dir: Optional[Path]) -> SplitStats:
    st = SplitStats(name=name)

    imgs = list_images(img_dir)
    st.total_images = len(imgs)

    if label_dir is None or not label_dir.exists():
        # no labels supplied for this split
        st.unlabeled_images = st.total_images
        return st

    label_files = {p.stem: p for p in label_dir.glob("*.txt")}
    for img in imgs:
        stem = img.stem
        txt = label_files.get(stem)

        if txt is None:
            st.images_missing_label_file += 1
            st.unlabeled_images += 1
            continue

        st.images_with_label_file += 1
        ok, n_boxes = parse_yolo_label_file(txt)
        if not ok:
            st.bad_label_files += 1
            if len(st.bad_label_examples) < 10:
                st.bad_label_examples.append(txt.name)
            # treat as missing usable annotation
            continue

        st.total_boxes += n_boxes
        if n_boxes == 0:
            st.healthy_images += 1
        else:
            st.infected_images += 1

    return st


def pct(part: int, whole: int) -> float:
    return 0.0 if whole == 0 else (part / whole) * 100.0


def print_table(headers: List[str], rows: List[List[str]]):
    # simple markdown-like table (no extra dependencies)
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(r):
        return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(r)) + " |"

    print(fmt_row(headers))
    print("| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |")
    for r in rows:
        print(fmt_row(r))


def summarize(stats: List[SplitStats], title: str):
    total_all = sum(s.total_images for s in stats)

    print(f"\n=== {title} ===")
    headers = ["Split", "Images", "% of used", "Has label file", "Missing label", "Healthy (0 box)", "Infected (>=1 box)", "Total boxes", "Bad label files"]
    rows = []
    for s in stats:
        rows.append([
            s.name,
            str(s.total_images),
            f"{pct(s.total_images, total_all):.2f}%",
            str(s.images_with_label_file),
            str(s.images_missing_label_file),
            str(s.healthy_images),
            str(s.infected_images),
            str(s.total_boxes),
            str(s.bad_label_files),
        ])
    print_table(headers, rows)

    # overall health distribution across labeled splits
    total_has_labels = sum(s.images_with_label_file for s in stats)
    total_healthy = sum(s.healthy_images for s in stats)
    total_infected = sum(s.infected_images for s in stats)

    print("\nOverall (for splits that have labels):")
    print(f"- Total images (all splits listed): {total_all}")
    print(f"- Total images with label files: {total_has_labels}")
    print(f"- Healthy (empty label): {total_healthy} ({pct(total_healthy, total_has_labels):.2f}%)")
    print(f"- Infected (has boxes): {total_infected} ({pct(total_infected, total_has_labels):.2f}%)")
    print(f"- Total bounding boxes: {sum(s.total_boxes for s in stats)}")

    bad_examples = []
    for s in stats:
        bad_examples.extend([f"{s.name}:{x}" for x in s.bad_label_examples])
    if bad_examples:
        print("\nBad label examples (first few):")
        for x in bad_examples[:10]:
            print(" -", x)


def main():
    # 1) Core splits you are using
    core = []

    core.append(analyze_split("Labeled-Train(GT)", LABELED_TRAIN_IMG, LABELED_TRAIN_LAB))
    core.append(analyze_split("Validation(GT)", LABELED_VAL_IMG, LABELED_VAL_LAB))

    # Test split is optional (only if folder exists)
    if TEST_IMG.exists():
        core.append(analyze_split("Test(GT)", TEST_IMG, TEST_LAB if TEST_LAB.exists() else None))

    # Unlabeled pool (by definition has no GT labels)
    if UNLABELED_IMG.exists():
        core.append(analyze_split("Unlabeled-Pool", UNLABELED_IMG, None))

    summarize(core, "DATASET USAGE SUMMARY (YOUR CURRENT EXPERIMENT)")

    # 2) Optional: audit pseudo-label directories (how many pseudo boxes created)
    if PSEUDO_DIRS:
        pseudo_stats = []
        for pdir in PSEUDO_DIRS:
            img_dir = pdir / "images"
            lab_dir = pdir / "labels"
            if pdir.exists():
                pseudo_stats.append(analyze_split(f"Pseudo:{pdir.name}", img_dir, lab_dir))
        if pseudo_stats:
            summarize(pseudo_stats, "PSEUDO-LABEL DIRECTORIES (OPTIONAL AUDIT)")

    print("\nDone ✅")
    print("Tip: Copy the tables into your thesis Dataset section (they’re already formatted).")


if __name__ == "__main__":
    main()
