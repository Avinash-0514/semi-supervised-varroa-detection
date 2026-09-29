from pathlib import Path
from ultralytics import YOLO

# ========= CONFIG Image path and Model =========
#TEACHER_WEIGHTS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\baseline_supervised_gpu3\weights\best.pt") # YOLO V8 weights
#TEACHER_WEIGHTS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\yolo11_baseline_supervised\weights\best.pt") # YOLO v11 weights
#================================================
#=========Config Image path and Model Experiment 2 =============
#TEACHER_WEIGHTS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\YOLOV8_Exp2_Supervised\weights\best.pt") # YOLO V8 weights
#TEACHER_WEIGHTS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\YOLOV11_Exp2_Supervised\weights\best.pt") # YOLO v11 weights
#TEACHER_WEIGHTS =  Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV8_Exp1_Supervised\weights\best.pt") # YOLO V8 weights Opt img 416
#TEACHER_WEIGHTS =  Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV11_Exp1_Supervised\weights\best.pt") # YOLO V11 weights Opt img 416

#TEACHER_WEIGHTS =  Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV8_Exp2_Supervised\weights\best.pt") # YOLO V8 weights Exp2 Opt img 416
TEACHER_WEIGHTS =  Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV11_Exp2_Supervised\weights\best.pt") # YOLO V11 weights Exp2 Opt img 416
#========== CONFIG Image path and Ablation Study Model Path==========

#TEACHER_WEIGHTS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\yolo8_Supervised_Ablation_Ratio_05\weights\best.pt") # Ablation 5 %
#TEACHER_WEIGHTS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\yolo8_Supervised_Ablation_Ratio_10\weights\best.pt") # Ablation 10 %
#TEACHER_WEIGHTS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\yolo8_Supervised_Ablation_Ratio_20\weights\best.pt") # Ablation 20 %
#TEACHER_WEIGHTS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\yolo8_Supervised_Ablation_Ratio_100\weights\best.pt") # Ablation 100%

# Unlabeled images folder (images only)
UNLABELED_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\unlabelled\images\all")
#==================================

# =========Where to save pseudo-labels (YOLO txt files v8)==============
#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov11_pseudo_conf025\label")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov11_pseudo_conf025\images")  # optional copy

#YOLO v8 25% Label Confidence

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov8_pseudo_conf025\label")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov8_pseudo_conf025\images")  # optional copy

#YOLO v8 50% Label Confidence

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov8_pseudo_conf050\label")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov8_pseudo_conf050\images")  # optional copy

#YOLO v8 65% Label Confidence

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov8_pseudo_conf065\label")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov8_pseudo_conf065\images")  # optional copy

#===============

#========== Where to Save pseudo-labels (YOLO txt files v11)================

#YOLO v11 65% Label Confidence

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov11_pseudo_conf065\label")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov11_pseudo_conf065\images")  # optional copy

#YOLO v11 50% Label Confidence
#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov11_pseudo_conf050\label")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov11_pseudo_conf050\images")  # optional copy

#YOLO v11 25% Label Confidence
#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLOv8_Ab_P\label")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Yolov11_pseudo_conf025\images")  # optional copy
#=============================================


#=========YOLO V8 25% Label Confidence Ablation Study=========

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLOv8_Ab5_Pseudo\label")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLOv8_Ab5_Pseudo\images")  # optional copy

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLOv8_Ab10_Pseudo\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLOv8_Ab10_Pseudo\images")  # optional copy

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLOv8_Ab20_Pseudo\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLOv8_Ab20_Pseudo\images")  # optional copy

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLOv8_Ab100_Pseudo\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLOv8_Ab100_Pseudo\images")  # optional copy

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Ablation_SSL\Conf_50\AB_RAW\YOLOv8_Ab5_Pseudo\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Ablation_SSL\Conf_50\AB_RAW\YOLOv8_Ab5_Pseudo\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Ablation_SSL\Conf_50\AB_RAW\YOLOv8_Ab10_Pseudo\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Ablation_SSL\Conf_50\AB_RAW\YOLOv8_Ab10_Pseudo\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Ablation_SSL\Conf_50\AB_RAW\YOLOv8_Ab20_Pseudo\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Ablation_SSL\Conf_50\AB_RAW\YOLOv8_Ab20_Pseudo\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Ablation_SSL\Conf_50\AB_RAW\YOLOv8_Ab100_Pseudo\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Ablation_SSL\Conf_50\AB_RAW\YOLOv8_Ab100_Pseudo\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Yolov8_pseudo_conf020\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Yolov8_pseudo_conf020\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Yolov8_pseudo_conf015\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Yolov8_pseudo_conf015\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Yolov8_pseudo_conf010\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Yolov8_pseudo_conf010\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11\Yolov11_pseudo_conf005\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11\Yolov11_pseudo_conf005\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11\Yolov11_pseudo_conf010\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11\Yolov11_pseudo_conf010\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11\Yolov11_pseudo_conf015\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11\Yolov11_pseudo_conf015\images")
#================= Experiment two =====================
#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf005\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf005\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf010\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf010\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf015\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf015\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf020\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf020\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf025\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf025\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf050\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf050\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf065\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08_v2\Yolov8_pseudo_conf065\images")


#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf005\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf005\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf010\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf010\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf015\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf015\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf020\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf020\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf005\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf005\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf050\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\\YOLO_11_v2\Yolov11_pseudo_conf050\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf065\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_11_v2\Yolov11_pseudo_conf065\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf005\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf005\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf010\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf010\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf015\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf015\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf020\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf020\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf025\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf025\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf050\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf050\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf065\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_08\Yolov08_pseudo_conf065\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf005_Exp2\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf005_Exp2\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf010_Exp2\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf010_Exp2\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf015_Exp2\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf015_Exp2\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf020_Exp2\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf020_Exp2\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf025_Exp2\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf025_Exp2\images")

#PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf050_Exp2\labels")
#PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf050_Exp2\images")

PSEUDO_LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf065_Exp2\labels")
PSEUDO_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Optimized\YOLO_V11\Yolov11_pseudo_conf065_Exp2\images")

CONF_THRES = 0.65
#CONF_THRES = 0.50  # start here
#CONF_THRES = 0.25 # Sweep Thres
#CONF_THRES = 0.20
#CONF_THRES = 0.15
#CONF_THRES = 0.10
#CONF_THRES = 0.05
IMGSZ = 416
DEVICE = 0  # GPU

# ==========================
def main():
    assert TEACHER_WEIGHTS.exists(), f"Teacher weights not found: {TEACHER_WEIGHTS}"
    assert UNLABELED_IMG_DIR.exists(), f"Unlabeled image dir not found: {UNLABELED_IMG_DIR}"

    PSEUDO_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    PSEUDO_IMG_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(TEACHER_WEIGHTS))

    # Get list of images
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in UNLABELED_IMG_DIR.iterdir() if p.suffix.lower() in exts])

    print(f"Teacher: {TEACHER_WEIGHTS}")
    print(f"Unlabeled images: {len(images)}")
    print(f"Confidence threshold: {CONF_THRES}")

    kept_total = 0
    empty_total = 0

    # Batch inference (Ultralytics handles batching internally if you pass a list)
    results = model.predict(
        source=[str(p) for p in images],
        conf=CONF_THRES,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
        stream=False
    )

    # Convert predictions to YOLO txt
    for img_path, r in zip(images, results):
        stem = img_path.stem
        out_txt = PSEUDO_LABEL_DIR / f"{stem}.txt"

        lines = []
        if r.boxes is not None and len(r.boxes) > 0:
            # boxes.xyxyn -> normalized x1 y1 x2 y2
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            # cls -> class id
            cls = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c in zip(xyxyn, cls):
                # YOLO format: class x_center y_center w h
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                w = (x2 - x1)
                h = (y2 - y1)
                lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            kept_total += len(lines)
        else:
            empty_total += 1

        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # Optional: copy images into pseudo/images for clean dataset packaging
        # (This avoids needing to reference unlabeled folder later)
        target_img = PSEUDO_IMG_DIR / img_path.name
        if not target_img.exists():
            target_img.write_bytes(img_path.read_bytes())

    print("\n✅ Pseudo-label generation complete")
    print(f"Pseudo label files written: {len(images)}")
    print(f"Total pseudo boxes kept: {kept_total}")
    print(f"Images with zero pseudo boxes: {empty_total}")
    print(f"Saved labels: {PSEUDO_LABEL_DIR}")
    print(f"Saved images: {PSEUDO_IMG_DIR}")

if __name__ == "__main__":
    main()
