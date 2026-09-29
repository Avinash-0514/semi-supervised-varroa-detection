import os
import csv
import cv2
from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================

VIDEO_PATH = r"C:\Users\User\Desktop\Thesis\Src_Yolo\VerroaMite.mp4"

MODEL_PATHS = [
    r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\YOLO_V8\Pseudo\yolo8_ssl_student_pseudo_conf025\weights\best.pt",
]

OUTPUT_DIR = r"C:\Users\User\Desktop\Thesis\Src_Yolo\VideoValidaiton"

CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45
IMG_SIZE = 1080
DEVICE = 0  

SAVE_CSV = True
SHOW_LIVE = True   # set False if you only want saved video
CLASS_NAMES = None  # keep None to read from model.names automatically


# =========================================================
# FUNCTION
# =========================================================

def run_video_inference(model_path, video_path, output_dir,
                        conf_threshold=0.25, iou_threshold=0.45,
                        imgsz=640, device=0, save_csv=True, show_live=True):

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    output_video_path = os.path.join(model_output_dir, f"{model_name}_annotated.mp4")
    csv_path = os.path.join(model_output_dir, f"{model_name}_detections.csv")

    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)

    names = model.names if hasattr(model, "names") else {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    csv_file = None
    csv_writer = None

    if save_csv:
        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_index",
            "time_sec",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ])

    frame_idx = 0
    total_detections = 0

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False
        )

        result = results[0]
        annotated_frame = frame.copy()

        frame_detection_count = 0

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                frame_detection_count += 1
                total_detections += 1

                # Draw rectangle
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Write CSV
                if csv_writer is not None:
                    time_sec = frame_idx / fps if fps > 0 else 0
                    csv_writer.writerow([
                        frame_idx,
                        round(time_sec, 3),
                        cls_id,
                        class_name,
                        round(conf, 4),
                        x1, y1, x2, y2
                    ])

        # Add frame info
        info_text = f"Frame: {frame_idx} | Detections: {frame_detection_count}"
        cv2.putText(
            annotated_frame,
            info_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        writer.write(annotated_frame)

        if show_live:
            cv2.imshow(f"Inference - {model_name}", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Stopped by user.")
                break

        frame_idx += 1

    cap.release()
    writer.release()

    if csv_file is not None:
        csv_file.close()

    if show_live:
        cv2.destroyAllWindows()

    print(f"\nFinished model: {model_name}")
    print(f"Annotated video saved to: {output_video_path}")
    if save_csv:
        print(f"CSV saved to: {csv_path}")
    print(f"Total detections: {total_detections}")


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_path in MODEL_PATHS:
        run_video_inference(
            model_path=model_path,
            video_path=VIDEO_PATH,
            output_dir=OUTPUT_DIR,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            device=DEVICE,
            save_csv=SAVE_CSV,
            show_live=SHOW_LIVE
        )


if __name__ == "__main__":
    main()