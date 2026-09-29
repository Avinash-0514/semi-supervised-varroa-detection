import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo")
SSL_ROOT = PROJECT_ROOT / "SSL"
RUNS_ROOT = PROJECT_ROOT / "runs" / "detect"

# Labeled
LAB_TRAIN_IMG = SSL_ROOT / "Labelled" / "images" / "train"
LAB_TRAIN_LAB = SSL_ROOT / "Labelled" / "labels" / "train"
LAB_VAL_IMG   = SSL_ROOT / "Labelled" / "images" / "val"
LAB_VAL_LAB   = SSL_ROOT / "Labelled" / "labels" / "val"

# Unlabeled images
UNLAB_IMG_DIR = SSL_ROOT / "unlabelled" / "images" / "all"

# Initial teacher weights = your supervised best model
#INIT_TEACHER = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\baseline_supervised_gpu3\weights\best.pt") # yolo_v8 Exp 1
#INIT_TEACHER = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\yolo11_baseline_supervised\weights\best.pt") # yolo_v11 Exp 1
#INIT_TEACHER = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\YOLOV8_Exp2_Supervised\weights\best.pt") # yolo_v8 Exp 2
#INIT_TEACHER = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\YOLOV11_Exp2_Supervised\weights\best.pt") # yolo_v11 Exp 2
#==============================
#Optimized Exp
#=========

#INIT_TEACHER = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV8_Exp1_Supervised\weights\best.pt") #yolo_v8 Exp1 Opt image 416
#INIT_TEACHER = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV11_Exp1_Supervised\weights\best.pt") #yolo_v11 Exp1 Opt image 416

#INIT_TEACHER = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV8_Exp2_Supervised\weights\best.pt") #yolo_v8 Exp2 Opt image 416
INIT_TEACHER = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV11_Exp2_Supervised\weights\best.pt") # Yolo_v11 Exp2 opt image 416
# Mean Teacher params
ROUNDS = 2            # start with 2 rounds (you can increase to 3 later)
EPOCHS_PER_ROUND = 20 # per round student training
CONF_THRES = 0.25     # we already found this works
IMGSZ = 416
DEVICE = 0
BATCH = 8
WORKERS = 4

EMA_DECAY = 0.99      # teacher smoothing (0.99 good start)
# ---------------------------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_all(src: Path, dst: Path, pattern: str):
    ensure_dir(dst)
    files = [f for f in src.rglob(pattern) if f.is_file()]  # ✅ recursive
    for f in files:
        shutil.copy2(f, dst / f.name)
    return len(files)


@torch.no_grad()
def ema_update_teacher(teacher_pt: Path, student_pt: Path, out_teacher_pt: Path, decay: float):
    """
    Safe EMA update using Ultralytics YOLO loader (avoids torch.load pickle issues in PyTorch 2.6).
    Teacher := decay*Teacher + (1-decay)*Student (for float tensors only).
    """
    teacher = YOLO(str(teacher_pt))
    student = YOLO(str(student_pt))

    t_state = teacher.model.state_dict()
    s_state = student.model.state_dict()

    for k in t_state.keys():
        if k in s_state and t_state[k].dtype.is_floating_point:
            t_state[k].mul_(decay).add_(s_state[k] * (1.0 - decay))

    teacher.model.load_state_dict(t_state, strict=False)

    # Save updated teacher weights
    teacher.save(str(out_teacher_pt))


def generate_pseudo_labels(teacher_weights: Path, out_pseudo_dir: Path):
    """
    Writes YOLO-format labels for every unlabeled image:
      out_pseudo_dir/images/*.jpg
      out_pseudo_dir/labels/*.txt  (may be empty for no objects)
    """
    out_img = out_pseudo_dir / "images"
    out_lab = out_pseudo_dir / "labels"
    ensure_dir(out_img)
    ensure_dir(out_lab)

    model = YOLO(str(teacher_weights))

    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in UNLAB_IMG_DIR.iterdir() if p.suffix.lower() in exts])

    kept = 0
    zero = 0

    results = model.predict(
        source=[str(p) for p in images],
        conf=CONF_THRES,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
        stream=True
    )

    for img_path, r in zip(images, results):
        # copy image
        dst_img = out_img / img_path.name
        if not dst_img.exists():
            dst_img.write_bytes(img_path.read_bytes())

        # label path
        out_txt = out_lab / f"{img_path.stem}.txt"
        lines = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c in zip(xyxyn, cls):
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                w = (x2 - x1)
                h = (y2 - y1)
                lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            kept += len(lines)
        else:
            zero += 1

        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"✅ Pseudo-labels created at {out_pseudo_dir}")
    print(f"   Unlabeled images: {len(images)} | boxes kept: {kept} | zero-box images: {zero}")


def build_student_dataset(pseudo_dir: Path, student_dir: Path):
    """
    Build:
      student_dir/images/train = labeled train + pseudo images
      student_dir/labels/train = labeled labels + pseudo labels
      student_dir/images/val   = labeled val
      student_dir/labels/val   = labeled val labels
    """
    img_train = student_dir / "images" / "train"
    lab_train = student_dir / "labels" / "train"
    img_val = student_dir / "images" / "val"
    lab_val = student_dir / "labels" / "val"

    # Fresh rebuild each round (clean)
    if student_dir.exists():
        shutil.rmtree(student_dir)
    ensure_dir(img_train); ensure_dir(lab_train); ensure_dir(img_val); ensure_dir(lab_val)

    # labeled train
    copy_all(LAB_TRAIN_IMG, img_train, "*.*")
    copy_all(LAB_TRAIN_LAB, lab_train, "*.txt")

    # pseudo train
    copy_all(pseudo_dir / "images", img_train, "*.*")
    copy_all(pseudo_dir / "labels", lab_train, "*.txt")

    # val
    copy_all(LAB_VAL_IMG, img_val, "*.*")
    copy_all(LAB_VAL_LAB, lab_val, "*.txt")

    print(f"✅ Student dataset rebuilt at {student_dir}")


def write_student_yaml(student_dir: Path, yaml_path: Path):
    yaml_text = f"""path: {SSL_ROOT.as_posix()}
train: {student_dir.relative_to(SSL_ROOT).as_posix()}/images/train
val: {student_dir.relative_to(SSL_ROOT).as_posix()}/images/val

nc: 1
names:
  - mite
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")


def train_student(student_yaml: Path, init_weights: Path, run_name: str):
    model = YOLO(str(init_weights))  # start from teacher (better than starting fresh)
    model.train(
        data=str(student_yaml),
        epochs=EPOCHS_PER_ROUND,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        name=run_name,
    )


def main():
    assert INIT_TEACHER.exists(), f"INIT_TEACHER not found: {INIT_TEACHER}"
    assert UNLAB_IMG_DIR.exists(), f"Unlabeled dir missing: {UNLAB_IMG_DIR}"

    teacher = PROJECT_ROOT / "mean_teacher" / "teacher_round0.pt"
    ensure_dir(teacher.parent)
    shutil.copy2(INIT_TEACHER, teacher)

    for r in range(1, ROUNDS + 1):
        print(f"\n====================")
        print(f"Mean Teacher Round {r}/{ROUNDS}")
        print(f"Teacher weights: {teacher}")
        print(f"====================\n")

        pseudo_dir = SSL_ROOT / f"pseudo_MT_round{r}"
        student_dir = SSL_ROOT / f"student_MT_round{r}"
        student_yaml = PROJECT_ROOT / f"student_MT_round{r}.yaml"

        # 1) Teacher -> pseudo labels
        generate_pseudo_labels(teacher, pseudo_dir)

        # 2) Build student dataset
        build_student_dataset(pseudo_dir, student_dir)
        write_student_yaml(student_dir, student_yaml)

        # 3) Train student starting from teacher weights
        run_name = f"Opt_yolo11_mean_teacher_Exp2_round{r}"
        train_student(student_yaml, teacher, run_name)

        # 4) Update teacher = EMA(teacher, student_best)
        student_best = RUNS_ROOT / run_name / "weights" / "best.pt"
        next_teacher = PROJECT_ROOT / "mean_teacher" / f"teacher_round{r}.pt"
        ema_update_teacher(teacher, student_best, next_teacher, EMA_DECAY)

        print(f"\n✅ Student best: {student_best}")
        print(f"✅ Updated teacher saved: {next_teacher}")

        teacher = next_teacher

    print("\n🎉 Mean Teacher rounds complete.")
    print(f"Final teacher weights: {teacher}")
    print("Next: evaluate final teacher or last student against baseline + your pseudo-label SSL run.")

if __name__ == "__main__":
    main()
