import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo")
SSL  = ROOT / "SSL"

# Ground-truth labeled
GT_TRAIN_IMG = SSL / "Labelled" / "images" / "train"
GT_TRAIN_LAB = SSL / "Labelled" / "labels" / "train"
GT_VAL_IMG   = SSL / "Labelled" / "images" / "val"
GT_VAL_LAB   = SSL / "Labelled" / "labels" / "val"

# Pseudo labels from YOLO11 teacher
PSEUDO_DIR   = SSL / "Yolov11_pseudo_conf025"
PSEUDO_IMG   = PSEUDO_DIR / "images"
PSEUDO_LAB   = PSEUDO_DIR / "labels"

# Output student dataset
OUT_DIR      = SSL / "student_yolo11_conf025"
OUT_TRAIN_IMG = OUT_DIR / "images" / "train"
OUT_TRAIN_LAB = OUT_DIR / "labels" / "train"
OUT_VAL_IMG   = OUT_DIR / "images" / "val"
OUT_VAL_LAB   = OUT_DIR / "labels" / "val"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_all(src: Path, dst: Path, pattern: str):
    ensure_dir(dst)
    files = list(src.glob(pattern))
    for f in files:
        shutil.copy2(f, dst / f.name)
    return len(files)

def sanity_check(img_dir: Path, lab_dir: Path):
    imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    labs = {p.stem for p in lab_dir.glob("*.txt")}
    print("\nSanity check:")
    print("Train images:", len(imgs))
    print("Train labels:", len(labs))
    print("Images without labels:", len(imgs - labs))
    print("Labels without images:", len(labs - imgs))

def main():
    # fresh rebuild
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    ensure_dir(OUT_TRAIN_IMG); ensure_dir(OUT_TRAIN_LAB)
    ensure_dir(OUT_VAL_IMG); ensure_dir(OUT_VAL_LAB)

    # Copy GT train
    gt_train_imgs = copy_all(GT_TRAIN_IMG, OUT_TRAIN_IMG, "*.*")
    gt_train_labs = copy_all(GT_TRAIN_LAB, OUT_TRAIN_LAB, "*.txt")

    # Copy pseudo train
    pseudo_imgs = copy_all(PSEUDO_IMG, OUT_TRAIN_IMG, "*.*")
    pseudo_labs = copy_all(PSEUDO_LAB, OUT_TRAIN_LAB, "*.txt")

    # Copy GT val
    val_imgs = copy_all(GT_VAL_IMG, OUT_VAL_IMG, "*.*")
    val_labs = copy_all(GT_VAL_LAB, OUT_VAL_LAB, "*.txt")

    print("✅ Student dataset created:")
    print(f"- labeled train images copied: {gt_train_imgs}")
    print(f"- labeled train labels copied: {gt_train_labs}")
    print(f"- pseudo images copied: {pseudo_imgs}")
    print(f"- pseudo labels copied: {pseudo_labs}")
    print(f"- val images copied: {val_imgs}")
    print(f"- val labels copied: {val_labs}")

    sanity_check(OUT_TRAIN_IMG, OUT_TRAIN_LAB)

if __name__ == "__main__":
    main()
