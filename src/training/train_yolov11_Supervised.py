from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")  # <-- YOLO11 nano
    model.train(
        data=r"C:\Users\User\Desktop\Thesis\Src_Yolo\YAML\SSL_data.yaml",  # your supervised labeled yaml
        epochs=50,
        imgsz=416,
        batch=8,
        device=0,
        workers=4,
        name="Opt_YOLOV11_Exp2_Supervised",
    )

if __name__ == "__main__":
    main()
