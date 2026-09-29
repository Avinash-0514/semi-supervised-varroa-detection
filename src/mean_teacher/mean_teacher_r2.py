import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo")
SSL_ROOT = PROJECT_ROOT / "SSL"
RUNS_ROOT = PROJECT_ROOT / "runs" / "detect"

TEACHER_ROUND0 = PROJECT_ROOT / "mean_teacher" / "teacher_round0.pt"
STUDENT_BEST_R1 = RUNS_ROOT / "yolo8_mean_teacher_round1" / "weights" / "best.pt"
TEACHER_ROUND1 = PROJECT_ROOT / "mean_teacher" / "teacher_round1.pt"

UNLAB_IMG_DIR = SSL_ROOT / "unlabelled" / "images" / "all"

CONF_THRES = 0.25
IMGSZ = 640
DEVICE = 0
BATCH = 8
WORKERS = 4
EPOCHS_ROUND2 = 20
EMA_DECAY = 0.99


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def ema_update_teacher(teacher_pt: Path, student_pt: Path, out_teacher_pt: Path, decay: float):
    teacher = YOLO(str(teacher_pt))
    student = YOLO(str(student_pt))
    t_state = teacher.model.state_dict()
    s_state = student.model.state_dict()

    for k in t_state.keys():
        if k in s_state and t_state[k].dtype.is_floating_point:
            t_state[k].mul_(decay).add_(s_state[k] * (1.0 - decay))

    teacher.model.load_state_dict(t_state, strict=False)
    teacher.save(str(out_teacher_pt))

def generate_pseudo_labels(teacher_weights: Path, out_pseudo_dir: Path):
    out_img = out_pseudo_dir / "images"
    out_lab = out_pseudo_dir / "labels"
    ensure_dir(out_img); ensure_dir(out_lab)

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
        dst_img = out_img / img_path.name
        if not dst_img.exists():
            dst_img.write_bytes(img_path.read_bytes())

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

    print(f"✅ Round2 pseudo-labels: images={len(images)} boxes={kept} zero_images={zero}")

def build_student_dataset(pseudo_dir: Path, student_dir: Path):
    import shutil

    lab_train_img = SSL_ROOT / "Labelled" / "images" / "train"
    lab_train_lab = SSL_ROOT / "Labelled" / "labels" / "train"
    lab_val_img   = SSL_ROOT / "Labelled" / "images" / "val"
    lab_val_lab   = SSL_ROOT / "Labelled" / "labels" / "val"

    img_train = student_dir / "images" / "train"
    lab_train = student_dir / "labels" / "train"
    img_val = student_dir / "images" / "val"
    lab_val = student_dir / "labels" / "val"

    if student_dir.exists():
        shutil.rmtree(student_dir)
    ensure_dir(img_train); ensure_dir(lab_train); ensure_dir(img_val); ensure_dir(lab_val)

    for f in lab_train_img.glob("*.*"): shutil.copy2(f, img_train / f.name)
    for f in lab_train_lab.glob("*.txt"): shutil.copy2(f, lab_train / f.name)

    for f in (pseudo_dir/"images").glob("*.*"): shutil.copy2(f, img_train / f.name)
    for f in (pseudo_dir/"labels").glob("*.txt"): shutil.copy2(f, lab_train / f.name)

    for f in lab_val_img.glob("*.*"): shutil.copy2(f, img_val / f.name)
    for f in lab_val_lab.glob("*.txt"): shutil.copy2(f, lab_val / f.name)

def write_student_yaml(student_dir: Path, yaml_path: Path):
    yaml_text = f"""path: {SSL_ROOT.as_posix()}
train: {student_dir.relative_to(SSL_ROOT).as_posix()}/images/train
val: {student_dir.relative_to(SSL_ROOT).as_posix()}/images/val

nc: 1
names:
  - mite
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")

def main():
    ensure_dir((PROJECT_ROOT / "mean_teacher"))

    # 1) Create teacher_round1 via EMA(teacher_round0, student_best_round1)
    print("Updating teacher_round1 via EMA...")
    ema_update_teacher(TEACHER_ROUND0, STUDENT_BEST_R1, TEACHER_ROUND1, EMA_DECAY)
    print("✅ Saved:", TEACHER_ROUND1)

    # 2) Round2 pseudo labels
    pseudo_dir = SSL_ROOT / "pseudo_MT_round2"
    generate_pseudo_labels(TEACHER_ROUND1, pseudo_dir)

    # 3) Build student dataset
    student_dir = SSL_ROOT / "student_MT_round2"
    build_student_dataset(pseudo_dir, student_dir)

    # 4) YAML and train student Round2
    student_yaml = PROJECT_ROOT / "student_MT_round2.yaml"
    write_student_yaml(student_dir, student_yaml)

    print("Training Mean Teacher Round2 student...")
    model = YOLO(str(TEACHER_ROUND1))
    model.train(
        data=str(student_yaml),
        epochs=EPOCHS_ROUND2,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        name="yolo8_mean_teacher_round2"
    )

if __name__ == "__main__":
    main()
