from ultralytics import YOLO

def main():
    m = YOLO(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV11_Exp2_Supervised\weights\best.pt")

    m.train(
        data=r"C:\Users\User\Desktop\Thesis\Src_Yolo\YAML\Opt_Yolov11_exp2\v11student_PL_065.yaml",
        epochs=50,
        imgsz=416,
        batch=8,
        device=0,
        workers=0,  
        name="Opt_yolo11_ssl_student_pseudo_conf065_EXP2"
    )
if __name__ == "__main__":
    main()  