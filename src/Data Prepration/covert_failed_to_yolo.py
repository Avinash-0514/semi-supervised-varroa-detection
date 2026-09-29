from pathlib import Path
from PIL import Image

# ====== CHANGE THIS to your SSL root folder ======
#SSL_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL")
SSL_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\New_labl")
# Your folders (based on what you used)
SPLITS = [
    {
        "name": "labeled_train",
        "images_dir": SSL_ROOT / "Labelled" / "images" / "train",
        "labels_dir": SSL_ROOT / "Labelled" / "labels" / "train",
        # backup created by your previous script run:
        "backup_dir": SSL_ROOT / "Labelled" / "labels" / "train_original_labeled_train",
    },
    {
        "name": "labeled_val",
        "images_dir": SSL_ROOT / "Labelled" / "images" / "val",
        "labels_dir": SSL_ROOT / "Labelled" / "labels" / "val",
        "backup_dir": SSL_ROOT / "Labelled" / "labels" / "val_original_labeled_val",
    },
    {
        "name": "test",
        "images_dir": SSL_ROOT / "test" / "images",
        "labels_dir": SSL_ROOT / "test" / "labels",
        "backup_dir": SSL_ROOT / "test" / "labels_original_test",
    },
]

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def find_image(images_dir: Path, stem: str) -> Path | None:
    # search recursively for any extension
    for p in images_dir.rglob(stem + ".*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    # fallback if extension is unusual
    for p in images_dir.glob(stem + ".*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None

def is_yolo_file(txt_path: Path) -> bool:
    """
    YOLO label file is either empty or each non-empty line has 5 tokens:
    class x_center y_center w h, with coords 0..1
    """
    content = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if content == "":
        return True  # empty = no objects is valid YOLO
    for ln in content.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            return False
        try:
            cls = int(float(parts[0]))
            vals = list(map(float, parts[1:]))
        except:
            return False
        if cls < 0:
            return False
        if any(v < 0.0 or v > 1.0 for v in vals):
            return False
    return True

def is_four_floats(line: str) -> bool:
    parts = line.split()
    if len(parts) != 4:
        return False
    try:
        list(map(float, parts))
        return True
    except:
        return False

def parse_custom_format(lines: list[str], label_name: str) -> list[tuple[float,float,float,float]]:
    """
    Supports:
    - '0'
    - 'N' then N lines of x1 y1 x2 y2 (but tolerates missing lines)
    - single line x1 y1 x2 y2 (no count)
    - multiple coord lines without count
    """
    lines = [ln.strip() for ln in lines if ln.strip() != ""]
    if not lines:
        return []

    # Case: only '0'
    if len(lines) == 1:
        try:
            v = float(lines[0])
            if int(v) == 0:
                return []
        except:
            pass

    # If first line looks like coords, treat it as coords directly
    if is_four_floats(lines[0]):
        boxes = []
        for ln in lines:
            if is_four_floats(ln):
                x1, y1, x2, y2 = map(float, ln.split())
                boxes.append((x1, y1, x2, y2))
        return boxes

    # Otherwise try parse first line as count
    try:
        n = int(float(lines[0]))
    except:
        # Unknown format: salvage any coord lines
        boxes = []
        for ln in lines:
            if is_four_floats(ln):
                x1, y1, x2, y2 = map(float, ln.split())
                boxes.append((x1, y1, x2, y2))
        return boxes

    if n <= 0:
        return []

    coord_lines = [ln for ln in lines[1:] if is_four_floats(ln)]
    if len(coord_lines) < n:
        print(f"⚠️ {label_name}: says {n} boxes but has {len(coord_lines)} coord lines. Using available lines.")

    boxes = []
    for ln in coord_lines[:min(n, len(coord_lines))]:
        x1, y1, x2, y2 = map(float, ln.split())
        boxes.append((x1, y1, x2, y2))
    return boxes

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def boxes_to_yolo_lines(boxes, w, h) -> list[str]:
    out = []
    for x1, y1, x2, y2 in boxes:
        xc = clamp01(((x1 + x2) / 2.0) / w)
        yc = clamp01(((y1 + y2) / 2.0) / h)
        bw = clamp01(abs(x2 - x1) / w)
        bh = clamp01(abs(y2 - y1) / h)
        out.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")  # class 0 = mite
    return out

def fix_split(split):
    name = split["name"]
    images_dir = split["images_dir"]
    labels_dir = split["labels_dir"]
    backup_dir = split["backup_dir"]

    print(f"\n=== FIXING: {name} ===")
    if not labels_dir.exists():
        print(f"Labels dir missing: {labels_dir}")
        return
    if not backup_dir.exists():
        print(f"Backup dir missing (can't safely fix): {backup_dir}")
        print("➡️ If you don't have backups, tell me and I’ll modify the script to parse current files instead.")
        return

    fixed = 0
    already_ok = 0
    no_image = 0
    missing_backup = 0
    failed = 0

    for lbl in sorted(labels_dir.glob("*.txt")):
        if is_yolo_file(lbl):
            already_ok += 1
            continue

        # use original content from backup (safer)
        backup_file = backup_dir / lbl.name
        if not backup_file.exists():
            missing_backup += 1
            continue

        img = find_image(images_dir, lbl.stem)
        if img is None:
            no_image += 1
            continue

        try:
            raw_lines = backup_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            boxes = parse_custom_format(raw_lines, lbl.name)
            with Image.open(img) as im:
                w, h = im.size
            yolo_lines = boxes_to_yolo_lines(boxes, w, h)
            lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            fixed += 1
        except Exception as e:
            failed += 1
            print(f"❌ Could not fix {lbl.name}: {e}")

    print(f"Already YOLO OK: {already_ok}")
    print(f"Fixed now: {fixed}")
    print(f"Missing backup: {missing_backup}")
    print(f"No matching image: {no_image}")
    print(f"Still failed: {failed}")

def main():
    for split in SPLITS:
        fix_split(split)

if __name__ == "__main__":
    main()
