from pathlib import Path
import pandas as pd

# Root where Ultralytics stores runs
RUNS_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\runs\detect")

# Add your runs here (you can extend this list later)
'''
EXPERIMENTS = [
    {
        "name": "baseline_supervised",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "baseline_supervised_gpu3",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised",
    },
    {
    "name": "yolo8_ssl_student_pseudo_conf025",
    "model": "YOLOv8n",
    "epochs": 50,
    "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf025_result",
    "best_by": "metrics/mAP50(B)",
    "setting": "SSL (Pseudo conf=0.25)",
    },
    {
    "name": "yolo8_ssl_student_pseudo_conf050",
    "model": "YOLOv8n",
    "epochs": 50,
    "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf050_result2",
    "best_by": "metrics/mAP50(B)",
    "setting": "SSL (Pseudo conf=0.50)",
    },
     {
    "name": "yolo8_ssl_student_pseudo_conf065",
    "model": "YOLOv8n",
    "epochs": 50,
    "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf065_result",
    "best_by": "metrics/mAP50(B)",
    "setting": "SSL (Pseudo conf=0.65)",
    },
    {
    "name": "yolo11_ssl_student_pseudo_conf025",
    "model": "YOLOv11n",
    "epochs": 50,
    "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf025_result",
    "best_by": "metrics/mAP50(B)",
    "setting": "SSL (Pseudo conf=0.25)",
    },
    {
    "name": "yolo11_ssl_student_pseudo_conf050",
    "model": "YOLOv11n",
    "epochs": 50,
    "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf050_result",
    "best_by": "metrics/mAP50(B)",
    "setting": "SSL (Pseudo conf=0.50)",
    },
    {
    "name": "yolo11_ssl_student_pseudo_conf065",
    "model": "YOLOv11n",
    "epochs": 50,
    "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf065_result",
    "best_by": "metrics/mAP50(B)",
    "setting": "SSL (Pseudo conf=0.65)",
    }
]
'''
'''
EXPERIMENTS = [
    {
        "name": "YOLOv8_SSL_Ablation_05",
        "model": "YOLOv8n_05%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_SSL_Ablation_Ratio_05",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Conf=0.25) Ablation 05 %",
    },
    {
        "name": "YOLOv8_SSL_Ablation_10",
        "model": "YOLOv8n_10%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_SSL_Ablation_Ratio_10",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Conf=0.25) Ablation 10 %",
    },
    {
        "name": "YOLOv8_SSL_Ablation_20",
        "model": "YOLOv8n_20%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_SSL_Ablation_Ratio_20",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Conf=0.25) Ablation 20 %",
    },
    {
        "name": "YOLOv8_SSL_Ablation_100",
        "model": "YOLOv8n_100%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_SSL_Ablation_Ratio_100",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Conf=0.25) Ablation 100 %",
    },
    {
        "name": "YOLOv8_Supervised_Ablation_05",
        "model": "YOLOv8n_05%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_Supervised_Ablation_Ratio_05",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised Ablation 05 %",
    },
    {
        "name": "YOLOv8_Supervised_Ablation_10",
        "model": "YOLOv8n_10%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_Supervised_Ablation_Ratio_10",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised Ablation 10 %",
    },
    {
        "name": "YOLOv8_Supervised_Ablation_20",
        "model": "YOLOv8n_20%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_Supervised_Ablation_Ratio_20",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised Ablation 20 %",
    },
    {
        "name": "YOLOv8_Supervised_Ablation_100",
        "model": "YOLOv8n_100%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_Supervised_Ablation_Ratio_100",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised Ablation 100 %",
    }
]


EXPERIMENTS = [
     {
        "name": "YOLOv8_Supervised_Ablation_05_50%",
        "model": "YOLOv8n_05%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_SSL_Ablation_Ratio_05_50%",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised Ablation 05 %",
    },
    {
        "name": "YOLOv8_Supervised_Ablation_10_50%",
        "model": "YOLOv8n_10%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_SSL_Ablation_Ratio_10_50%",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised Ablation 10 %",
    },
    {
        "name": "YOLOv8_Supervised_Ablation_20_50%",
        "model": "YOLOv8n_20%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_SSL_Ablation_Ratio_20_50%",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised Ablation 20 %",
    },
    {
        "name": "YOLOv8_Supervised_Ablation_100_50%",
        "model": "YOLOv8n_100%",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_SSL_Ablation_Ratio_100_50%",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised Ablation 100 %",
    },

]

EXPERIMENTS =[
    {
        "name": "yolo8_ssl_student_pseudo_conf005%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf005",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.05)",
    },
    {
        "name": "yolo8_ssl_student_pseudo_conf010%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf010",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.10)",
    },
    {
        "name": "yolo8_ssl_student_pseudo_conf015%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf015",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.15)",
    },
    {
        "name": "yolo8_ssl_student_pseudo_conf020%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf020",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.20)",
    }
]

EXPERIMENTS =[
    {
        "name": "yolo8_ssl_student_pseudo_conf005%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf005_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.05)",
    },
    {
        "name": "yolo8_ssl_student_pseudo_conf010%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf010_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.10)",
    },
    {
        "name": "yolo8_ssl_student_pseudo_conf015%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf015_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.15)",
    },
    {
        "name": "yolo8_ssl_student_pseudo_conf020%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf020_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.20)",
    },
    {
        "name": "yolo8_ssl_student_pseudo_conf025%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf025_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.25)",
    },
    {
        "name": "yolo8_ssl_student_pseudo_conf050%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf050_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.50)",
    },
    {
        "name": "yolo8_ssl_student_pseudo_conf065%",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo8_ssl_student_pseudo_conf065_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.65)",
    },
    {
        "name": "yolo8_BaseLine_Supervised_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLOV8_Exp2_Supervised",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL Supervised",
    },

]

EXPERIMENTS =[
    {
        "name": "yolo11_ssl_student_pseudo_conf005%",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf005_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.05)",
    },
    {
        "name": "yolo11_ssl_student_pseudo_conf010%",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf010_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.10)",
    },
    {
        "name": "yolo11_ssl_student_pseudo_conf015%",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf015_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.15)",
    },
    {
        "name": "yolo11_ssl_student_pseudo_conf020%",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf020_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.20)",
    },
    {
        "name": "yolo11_ssl_student_pseudo_conf025%",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf025_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.25)",
    },
    {
        "name": "yolo11_ssl_student_pseudo_conf050%",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf050_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.50)",
    },
    {
        "name": "yolo11_ssl_student_pseudo_conf065%",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "yolo11_ssl_student_pseudo_conf065_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL (Pseudo conf=0.65)",
    },
    {
        "name": "yolo11_BaseLine_Supervised_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLOV11_Exp2_Supervised",
        "best_by": "metrics/mAP50(B)",
        "setting": "SSL Supervised",
    },

]

EXPERIMENTS =[
    {
        "name": "yolo8_ssl_Mean_Teacher_25%_Exp1_R1",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLO_V8"/"Mean_Teacher"/"yolo8_mean_teacher_round2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Mean Teacher conf = 25%",
    },
    {
        "name": "yolo8_ssl_Mean_Teacher_25%_Exp1_R1",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLO_V8"/"Mean_Teacher"/"yolo8_mean_teacher_round1",
        "best_by": "metrics/mAP50(B)",
        "setting": "Mean Teacher conf = 25%",
    },
    {
        "name": "yolo8_ssl_Mean_Teacher_25%_Exp2_R2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLO_V8"/"Mean_Teacher"/"yolo8_mean_teacher_Exp2_round2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Mean Teacher conf = 25%",
    },
    {
        "name": "yolo8_ssl_Mean_Teacher_25%_Exp2_R1",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLO_V8"/"Mean_Teacher"/"yolo8_mean_teacher_Exp2_round1",
        "best_by": "metrics/mAP50(B)",
        "setting": "Mean Teacher conf = 25%",
    },
     {
        "name": "yolo11_ssl_Mean_Teacher_25%_Exp1_R2",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLO_V11"/"Mean_Teacher"/"yolo11_mean_teacher_round2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Mean Teacher conf = 25%",
    },
    {
        "name": "yolo11_ssl_Mean_Teacher_25%_Exp1_R1",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLO_V11"/"Mean_Teacher"/"yolo11_mean_teacher_round1",
        "best_by": "metrics/mAP50(B)",
        "setting": "Mean Teacher conf = 25%",
    },
    {
        "name": "yolo11_ssl_Mean_Teacher_25%_Exp2_R2",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLO_V11"/"Mean_Teacher"/"yolo11_mean_teacher_Exp2_round2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Mean Teacher conf = 25%",
    },
    {
        "name": "yolo11_ssl_Mean_Teacher_25%_Exp2_R1",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "YOLO_V11"/"Mean_Teacher"/"yolo11_mean_teacher_Exp2_round1",
        "best_by": "metrics/mAP50(B)",
        "setting": "Mean Teacher conf = 25%",
    },
]

EXPERIMENTS =[
    
        {
        "name": "Opt_yolo8_SSL_05%_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_ssl_student_pseudo_conf005_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 05%",
         },
        {
        "name": "Opt_yolo8_SSL_10%_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_ssl_student_pseudo_conf010_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 10%",
        },
        {
        "name": "Opt_yolo8_SSL_15%_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_ssl_student_pseudo_conf015_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 15%",
        },
        {
        "name": "Opt_yolo8_SSL_20%_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_ssl_student_pseudo_conf020_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 20%",
        },
        {
        "name": "Opt_yolo8_SSL_25%_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_ssl_student_pseudo_conf025_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 25%",
        },
        {
        "name": "Opt_yolo8_SSL_50%_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_ssl_student_pseudo_conf050_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 50%",
        },
        {
        "name": "Opt_yolo8_SSL_65%_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_ssl_student_pseudo_conf065_EXP2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 65%",
        
        }

]

EXPERIMENTS =[
    {
        "name": "Opt_yolo8_SSL_25%_Exp1",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_mean_teacher_Exp1_round1",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 25%",   
    },
    {
        "name": "Opt_yolo8_SSL_25%_Exp1",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_mean_teacher_Exp1_round2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 25%",   
    },
    {
        "name": "Opt_yolo8_SSL_25%_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_mean_teacher_Exp2_round1",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 25%",   
    },
    {
        "name": "Opt_yolo8_SSL_25%_Exp2",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo8_mean_teacher_Exp2_round2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V8 Conf = 25%",   
    },
    {
        "name": "Opt_yolo11_SSL_25%_Exp1",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo11_mean_teacher_Exp1_round1",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V11 Conf = 25%",   
    },
    {
        "name": "Opt_yolo11_SSL_25%_Exp1",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo11_mean_teacher_Exp1_round2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V11 Conf = 25%",   
    },
    {
        "name": "Opt_yolo11_SSL_25%_Exp2",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo11_mean_teacher_Exp2_round1",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V11 Conf = 25%",   
    },
    {
        "name": "Opt_yolo11_SSL_25%_Exp2",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_yolo11_mean_teacher_Exp2_round2",
        "best_by": "metrics/mAP50(B)",
        "setting": "Opt_YOLO_V11 Conf = 25%",   
    },


]
'''
EXPERIMENTS =[
    {
        "name": "Opt_Yolov8_Exp1_Supervised",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_YOLOV8_Exp1_Supervised",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised",   
    },
    {
        "name": "Opt_Yolov8_Exp2_Supervised",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_YOLOV8_Exp2_Supervised",
        "best_by": "metrics/mAP50(B)",
        "setting":"Supervised",   
    },
    {
        "name": "Opt_Yolov11_Exp1_Supervised",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_YOLOV11_Exp1_Supervised",
        "best_by": "metrics/mAP50(B)",
        "setting": "Supervised",   
    },
    {
        "name": "Opt_Yolov11_Exp2_Supervised",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT / "Opt_YOLOV11_Exp2_Supervised",
        "best_by": "metrics/mAP50(B)",
        "setting":"Supervised",   
    },
    {
        "name": "Yolov08_Exp1_Supervised",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT/ "BaselineModel" / "baseline_supervised_gpu3",
        "best_by": "metrics/mAP50(B)",
        "setting":"Supervised",   
    },
    {
        "name": "Yolov08_Exp2_Supervised",
        "model": "YOLOv8n",
        "epochs": 50,
        "run_dir": RUNS_ROOT/ "BaselineModel" / "YOLOV8_Exp2_Supervised",
        "best_by": "metrics/mAP50(B)",
        "setting":"Supervised",   
    },
    {
        "name": "Yolov11_Exp1_Supervised",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT/ "BaselineModel" / "yolo11_baseline_supervised",
        "best_by": "metrics/mAP50(B)",
        "setting":"Supervised",   
    },
    {
        "name": "Yolov11_Exp2_Supervised",
        "model": "YOLOv11n",
        "epochs": 50,
        "run_dir": RUNS_ROOT/ "BaselineModel" / "YOLOV11_Exp2_Supervised",
        "best_by": "metrics/mAP50(B)",
        "setting":"Supervised",   
    }
]

def extract_best(run_dir: Path, best_by: str):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results.csv: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if best_by not in df.columns:
        raise KeyError(f"Column '{best_by}' not found in {csv_path}. Columns: {list(df.columns)}")

    best_idx = df[best_by].astype(float).idxmax()
    best = df.loc[best_idx]

    return {
        "BestEpoch": int(best["epoch"]),
        "Precision": float(best["metrics/precision(B)"]),
        "Recall": float(best["metrics/recall(B)"]),
        "mAP@0.5": float(best["metrics/mAP50(B)"]),
        "mAP@0.5:0.95": float(best["metrics/mAP50-95(B)"]),
    }

def main():
    rows = []
    for exp in EXPERIMENTS:
        best = extract_best(exp["run_dir"], exp["best_by"])
        rows.append({
            "Experiment": exp["name"],
            "Setting": exp["setting"],
            "Model": exp["model"],
            "Epochs": exp["epochs"],
            "BestEpoch": best["BestEpoch"],
            "Precision": best["Precision"],
            "Recall": best["Recall"],
            "mAP@0.5": best["mAP@0.5"],
            "mAP@0.5:0.95": best["mAP@0.5:0.95"],
            "BestBy": exp["best_by"],
            "RunDir": str(exp["run_dir"]),
        })

    out_df = pd.DataFrame(rows).sort_values(by=["Setting", "mAP@0.5"], ascending=[True, False])

    out_csv = RUNS_ROOT / "comparison_table.csv"
    out_md = RUNS_ROOT / "comparison_table.md"

    out_df.to_csv(out_csv, index=False)

    # Markdown table
    md = out_df[["Experiment","Setting","Model","Epochs","BestEpoch","Precision","Recall","mAP@0.5","mAP@0.5:0.95"]].copy()
    md["Precision"] = md["Precision"].map(lambda x: f"{x:.3f}")
    md["Recall"] = md["Recall"].map(lambda x: f"{x:.3f}")
    md["mAP@0.5"] = md["mAP@0.5"].map(lambda x: f"{x:.3f}")
    md["mAP@0.5:0.95"] = md["mAP@0.5:0.95"].map(lambda x: f"{x:.3f}")

    out_md.write_text(md.to_markdown(index=False), encoding="utf-8")

    print("✅ Saved:")
    print("-", out_csv)
    print("-", out_md)
    print("\n=== Preview ===")
    print(md.to_string(index=False))

if __name__ == "__main__":
    main()
