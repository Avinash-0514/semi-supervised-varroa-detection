from pathlib import Path
from ultralytics import YOLO

TEACHER_WEIGHTS = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\baseline_supervised_gpu3\weights\best.pt")
UNLABELED_IMG_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\unlabelled\images\all")

IMGSZ = 320
DEVICE = 0
THRESHOLDS = [0.25, 0.35, 0.50]

def main():
    model = YOLO(str(TEACHER_WEIGHTS))
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in UNLABELED_IMG_DIR.iterdir() if p.suffix.lower() in exts])

    print(f"Unlabeled images: {len(images)}")
    print(f"Teacher: {TEACHER_WEIGHTS}\n")

    for conf in THRESHOLDS:
        kept_boxes = 0
        zero_images = 0

        results = model.predict(
            source=[str(p) for p in images],
            conf=conf,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False,
            stream=True  # IMPORTANT
        )

        # stream=True returns a generator
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                kept_boxes += len(r.boxes)
            else:
                zero_images += 1

        print(f"conf={conf:.2f}  ->  boxes={kept_boxes}   zero_images={zero_images}")

if __name__ == "__main__":
    main()
