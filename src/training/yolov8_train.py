from ultralytics import YOLO

def main():
    DATA_YAML = r"C:\Users\User\Desktop\Thesis\Src_Yolo\YAML\SSL_data.yaml"
    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=416,
        batch=8,
        device=0,
        workers=4,
        name="Opt_YOLOV8_Exp2_Supervised",
    )

if __name__ == "__main__":
    main()
