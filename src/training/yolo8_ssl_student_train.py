from ultralytics import YOLO

def main():
    #model = YOLO("yolov8n.pt")  # start from pretrained
    model = YOLO(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV8_Exp2_Supervised\weights\best.pt")
    model.train(
        data = r"C:\Users\User\Desktop\Thesis\Src_Yolo\YAML\Opt_YoloV8_exp2\v8student_PL_065.yaml", 
        epochs=50,
        imgsz=416,
        batch=8,
        device=0,
        workers=4,
        name="Opt_yolo8_ssl_student_pseudo_conf065_EXP2",
    )

if __name__ == "__main__":
    main()
