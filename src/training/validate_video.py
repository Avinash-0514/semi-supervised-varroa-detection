
from ultralytics import YOLO
from pathlib import Path
import os

def main():
    # =========================
    # Model Path
    # =========================

    #Supervised Model

    #model_path =r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV8_Exp1_Supervised\weights\best.pt" # Supervised Model Yolo v8 opt Exp1
    #model_path =r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV8_Exp2_Supervised\weights\best.pt"  # Supervised Model Yolo v8 opt Exp2
    #model_path =r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV11_Exp1_Supervised\weights\best.pt" # Supervised Model Yolo v11 opt Exp1
    #model_path =r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_YOLOV11_Exp2_Supervised\\weights\best.pt" # Supervised Model Yolo v11 opt Exp2
    #****Mean Teacher path ****
    
    #model_path = r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_yolo8_mean_teacher_Exp1_round1\weights\best.pt" # MT Exp 1 YOLO 8
    #model_path = r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_yolo8_mean_teacher_Exp2_round1\weights\best.pt" # MT Exp 2 YOLO 8

    #model_path = r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_yolo11_mean_teacher_Exp1_round1\weights\best.pt" # MT Exp 1 YOLO 11
    #model_path =r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_yolo11_mean_teacher_Exp2_round1\weights\best.pt" # MT Exp 2 YOLO 11

    #****Pseudo Label Model Paths*****
    #YOLO V08

    #model_path = r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_yolo8_ssl_student_pseudo_conf025_EXP1\weights\best.pt" # SSL Exp 1 YOLO 8
    model_path = r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_yolo8_ssl_student_pseudo_conf025_EXP2\weights\best.pt" #SSL EXP 2 YOLO 8
    
    
    #YOLO V11
    #model_path = r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_yolo11_ssl_student_pseudo_conf025_EXP1\\weights\best.pt" # SSL Exp 1 YOLO 11
    #model_path = r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect\Opt_yolo11_ssl_student_pseudo_conf025_EXP2\\weights\best.pt" # SSL Exp 2 YOLO 11
    #========================
    #YAML Path for Data
    #========================
    #data_yaml =r"C:\Users\User\Desktop\Thesis\Src_Yolo\YAML\VideoValidation\video_val_easyv1.yaml"
    #data_yaml =r"C:\Users\User\Desktop\Thesis\Src_Yolo\YAML\VideoValidation\video_val_mediumv1.yaml"
    #data_yaml =r"C:\Users\User\Desktop\Thesis\Src_Yolo\YAML\VideoValidation\video_val_hardv1.yaml"
    data_yaml = r"C:\Users\User\Desktop\Thesis\Src_Yolo\YAML\VideoValidation\video_val_hardv2.yaml"
    #data_yaml = r"C:\Users\User\Desktop\Thesis\Src_Yolo\YAML\VideoValidation\video_val_mediumv2.yaml"
    #========================
    # Frame Images Path
    #========================

    #image_source = r"C:\Users\User\Desktop\Thesis\Src_Yolo\Video Validation_Test\images"
    #image_source = r"C:\Users\User\Desktop\Thesis\Src_Yolo\Video Validation_Test\V1_Medium\images"
    #image_source = r"C:\Users\User\Desktop\Thesis\Src_Yolo\Video Validation_Test\V1_Hard\images"
    image_source = r"C:\Users\User\Desktop\Thesis\Src_Yolo\Video Validation_Test\V2_Hard\images"
    #image_source = r"C:\Users\User\Desktop\Thesis\Src_Yolo\Video Validation_Test\V2_Medium\images"
    #image_source = r"C:\Users\User\Desktop\Thesis\Src_Yolo\Video Validation_Test\V1_Easy\images"

    val_imgsz = 1280      # try 640, 960, 1280
    pred_imgsz = 960     # keep same for fair check
    conf_thres = 0.10    # try 0.10, 0.15, 0.25
    iou_thres = 0.45



    # =========================
    # 2. LOAD MODEL
    # =========================
    model = YOLO(model_path)

 # =========================
    # Inference Settings
    # =========================
    val_imgsz = 1280      # try 640, 960, 1280
    pred_imgsz = 960     # keep same for fair check
    conf_thres = 0.10    # try 0.10, 0.15, 0.25
    iou_thres = 0.45

    # =========================
    # Load Model
    # =========================
    model = YOLO(model_path)

    # =========================
    # Validate
    # =========================
  
    metrics = model.val(
    data=data_yaml,
    split="val",
    imgsz=960,
    conf=0.001,
    iou=0.01,
    plots=True,
    workers=0,
    device=0,
    augment=False
    )
    # =========================
    # Print Results
    # =========================
    print("\n========== VIDEO VALIDATION RESULTS ==========")
    print(f"Precision      : {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall         : {metrics.results_dict['metrics/recall(B)']:.4f}")
    print(f"mAP@0.5        : {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95   : {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"Fitness        : {metrics.results_dict['fitness']:.4f}")
    print(model.names)
    print("=============================================\n")

    # =========================
    # Visual Prediction Check
    # =========================
    model.predict(
        source=image_source,
        imgsz=pred_imgsz,
        conf=conf_thres,
        iou=iou_thres,
        augment=True,
        save=True,
        device=0,
        project=r"C:/Users/User/Desktop/Thesis/Src_Yolo/runs/detect",
        name="Prediction_SSL_Yolo8_Exp2_Opt_Hard",
        exist_ok=True
    )

    print("Prediction images saved successfully.")

if __name__ == "__main__":
    main()