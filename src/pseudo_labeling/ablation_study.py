# build_labeled_ratio_subsets.py
from pathlib import Path
import random
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(d: Path):
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS])

def count_boxes(lbl: Path) -> int:
    if not lbl.exists():
        return -1  # missing
    txt = lbl.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return 0
    return len([ln for ln in txt.splitlines() if ln.strip()])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def write_yaml(yaml_path: Path, train_images: Path, val_images: Path, names=("mite",)):
    lines = [
        f"path: {yaml_path.parent.as_posix()}",
        f"train: {train_images.as_posix()}",
        f"val: {val_images.as_posix()}",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def stratified_pick(items, ratio, seed):
    rng = random.Random(seed)
    infected = [x for x in items if x["infected"]]
    healthy  = [x for x in items if not x["infected"]]

    n_total = len(items)
    n_take = max(1, int(round(n_total * ratio)))

    inf_ratio = len(infected) / n_total if n_total else 0
    n_inf = min(len(infected), int(round(n_take * inf_ratio)))
    n_healthy = min(len(healthy), n_take - n_inf)

    rng.shuffle(infected); rng.shuffle(healthy)
    picked = infected[:n_inf] + healthy[:n_healthy]
    rng.shuffle(picked)
    return picked

def main():
    # ✅ CHANGE THESE PATHS ONLY
    GT_TRAIN_IMAGES = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Labelled\images\train")
    GT_TRAIN_LABELS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Labelled\labels\train")

    GT_VAL_IMAGES   = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Labelled\images\val")
    GT_VAL_LABELS   = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Labelled\labels\val")

    OUT_ROOT        = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\ablation")
    RATIOS          = [0.05, 0.10, 0.20, 1.00]
    SEED            = 42
    CLASS_NAMES     = ("mite",)

    # Build item list from GT labeled train
    items = []
    for img in list_images(GT_TRAIN_IMAGES):
        lbl = GT_TRAIN_LABELS / (img.stem + ".txt")
        n = count_boxes(lbl)
        if n < 0:
            continue  # skip missing labels
        items.append({"img": img, "lbl": lbl, "infected": (n > 0), "boxes": n})

    total = len(items)
    inf = sum(1 for x in items if x["infected"])
    healthy = total - inf
    print(f"GT labeled train images with label files: {total} (infected={inf}, healthy={healthy})")

    # Prepare val copy once per ratio (we copy into each ratio folder for simplicity)
    val_imgs = list_images(GT_VAL_IMAGES)

    for r in RATIOS:
        ratio_name = f"ratio_{int(r*100):02d}"
        ds = OUT_ROOT / ratio_name
        tr_img_out = ds / "images" / "train"
        tr_lbl_out = ds / "labels" / "train"
        va_img_out = ds / "images" / "val"
        va_lbl_out = ds / "labels" / "val"

        ensure_dir(tr_img_out); ensure_dir(tr_lbl_out)
        ensure_dir(va_img_out); ensure_dir(va_lbl_out)

        picked = items if r >= 0.999 else stratified_pick(items, r, SEED)
        p_inf = sum(1 for x in picked if x["infected"])
        print(f"\n[{ratio_name}] picked={len(picked)} infected={p_inf} healthy={len(picked)-p_inf}")

        # Copy picked train subset
        for x in picked:
            copy_file(x["img"], tr_img_out / x["img"].name)
            copy_file(x["lbl"], tr_lbl_out / x["lbl"].name)

        # Copy full val
        for img in val_imgs:
            lbl = GT_VAL_LABELS / (img.stem + ".txt")
            if not lbl.exists():
                continue
            copy_file(img, va_img_out / img.name)
            copy_file(lbl, va_lbl_out / lbl.name)

        # Write yaml
        yaml_path = ds / "data.yaml"
        write_yaml(yaml_path, tr_img_out, va_img_out, CLASS_NAMES)
        print(f"✅ Created: {ds}")
        print(f"✅ YAML: {yaml_path}")

    print("\nDone ✅ Now train each ratio by pointing your training .py to the YAML.")

if __name__ == "__main__":
    main()
