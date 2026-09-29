from ultralytics import YOLO

model = YOLO(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_yolo8_ssl_student_pseudo_conf025_EXP1\weights\best.pt")

# One segment
model.predict(
    source=r"C:\Users\User\Desktop\Thesis\Src_Yolo\Picked_video\V1_segment_000_easy.mp4",
    save=True,
    name="segment_hard_result"
)