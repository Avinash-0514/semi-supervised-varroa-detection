from pathlib import Path
import shutil

SSL_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL")
#SSL_AB_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\YOLO_08\Ablation_SSL\Conf_50\AB_RAW")

# Human labeled data
LABELED_TRAIN_IMG = SSL_ROOT / "Labelled" / "images" / "train"
LABELED_TRAIN_LAB = SSL_ROOT / "Labelled" / "labels" / "train"
LABELED_VAL_IMG   = SSL_ROOT / "Labelled" / "images" / "val"
LABELED_VAL_LAB   = SSL_ROOT / "Labelled" / "labels" / "val"

# Ablation Data
# 5 % Ratio

#LABELED_TRAIN_IMG = SSL_ROOT / "ablation" /"ratio_05" / "images" / "train"
#LABELED_TRAIN_LAB = SSL_ROOT / "ablation" /"ratio_05" / "labels" / "train"
#LABELED_VAL_IMG   = SSL_ROOT / "ablation" /"ratio_05" / "images" / "val"
#LABELED_VAL_LAB   = SSL_ROOT / "ablation" /"ratio_05" / "labels" / "val"

#10 % Ratio

#LABELED_TRAIN_IMG = SSL_ROOT / "ablation" /"ratio_10" / "images" / "train"
#LABELED_TRAIN_LAB = SSL_ROOT / "ablation" /"ratio_10" / "labels" / "train"
#LABELED_VAL_IMG   = SSL_ROOT / "ablation" /"ratio_10" / "images" / "val"
#LABELED_VAL_LAB   = SSL_ROOT / "ablation" /"ratio_10" / "labels" / "val"

#20 % Ratio

#LABELED_TRAIN_IMG = SSL_ROOT / "ablation" /"ratio_20" / "images" / "train"
#LABELED_TRAIN_LAB = SSL_ROOT / "ablation" /"ratio_20" / "labels" / "train"
#LABELED_VAL_IMG   = SSL_ROOT / "ablation" /"ratio_20" / "images" / "val"
#LABELED_VAL_LAB   = SSL_ROOT / "ablation" /"ratio_20" / "labels" / "val"

#100 % Ratio

#LABELED_TRAIN_IMG = SSL_ROOT / "ablation" /"ratio_100" / "images" / "train"
#LABELED_TRAIN_LAB = SSL_ROOT / "ablation" /"ratio_100" / "labels" / "train"
#LABELED_VAL_IMG   = SSL_ROOT / "ablation" /"ratio_100" / "images" / "val"
#LABELED_VAL_LAB   = SSL_ROOT / "ablation" /"ratio_100" / "labels" / "val"

#YOLO v8 Student labelling
#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.65)
#PSEUDO_IMG = SSL_ROOT/ "YOLO_08_v2" / "Yolov8_pseudo_conf065" / "images"
#PSEUDO_LAB = SSL_ROOT/ "YOLO_08_v2" / "Yolov8_pseudo_conf065" / "labels"
# Student output dataset yolo v8 conf 0.65
#STUDENT_IMG_TRAIN = SSL_ROOT/ "YOLO_08_v2" / "student_yolo_v8_conf065" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT/ "YOLO_08_v2" / "student_yolo_v8_conf065" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT/ "YOLO_08_v2" / "student_yolo_v8_conf065" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT/ "YOLO_08_v2" / "student_yolo_v8_conf065" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.50)
#PSEUDO_IMG = SSL_ROOT/ "YOLO_08_v2" / "Yolov8_pseudo_conf050" / "images"
#PSEUDO_LAB = SSL_ROOT/ "YOLO_08_v2" / "Yolov8_pseudo_conf050" / "labels"
# Student output dataset yolo v8 conf 0.50
#STUDENT_IMG_TRAIN = SSL_ROOT/ "YOLO_08_v2" / "student_yolo_v8_conf050" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT/ "YOLO_08_v2" / "student_yolo_v8_conf050" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT/ "YOLO_08_v2" / "student_yolo_v8_conf050" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT/ "YOLO_08_v2" / "student_yolo_v8_conf050" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.25)
#PSEUDO_IMG = SSL_ROOT  / "YOLO_08_v2"/ "Yolov8_pseudo_conf025" / "images"
#PSEUDO_LAB = SSL_ROOT  / "YOLO_08_v2"/ "Yolov8_pseudo_conf025" / "labels"
# Student output dataset yolo v8 conf 0.25
#STUDENT_IMG_TRAIN = SSL_ROOT  / "YOLO_08_v2"/ "student_yolo_v8_conf025" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT  / "YOLO_08_v2"/ "student_yolo_v8_conf025" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT  / "YOLO_08_v2"/ "student_yolo_v8_conf025" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT  / "YOLO_08_v2"/ "student_yolo_v8_conf025" / "labels" / "val"
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.20)
#PSEUDO_IMG = SSL_ROOT / "YOLO_08_v2"/"Yolov8_pseudo_conf020" / "images"
#PSEUDO_LAB = SSL_ROOT /  "YOLO_08_v2"/"Yolov8_pseudo_conf020" / "labels"
# Student output dataset yolo v8 conf 0.20
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf020" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf020" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf020" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf020" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.15)
#PSEUDO_IMG = SSL_ROOT / "YOLO_08_v2"/"Yolov8_pseudo_conf015" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_08_v2"/"Yolov8_pseudo_conf015" / "labels"
# Student output dataset yolo v8 conf 0.15
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf015" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf015" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf015" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf015" / "labels" / "val"
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.10
#PSEUDO_IMG = SSL_ROOT / "YOLO_08_v2"/"Yolov8_pseudo_conf010" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_08_v2"/"Yolov8_pseudo_conf010" / "labels"
# Student output dataset yolo v8 conf 0.10
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf010" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf010" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf010" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf010" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.05
#PSEUDO_IMG = SSL_ROOT / "YOLO_08_v2"/"Yolov8_pseudo_conf005" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_08_v2"/"Yolov8_pseudo_conf005" / "labels"
# Student output dataset yolo v8 conf 0.05
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf005" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf005" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf005" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_08_v2"/"student_yolo_v8_conf005" / "labels" / "val"
#---------------------------------------------------------------------------

#YOLOV11 Experiment two ==============

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.05
#PSEUDO_IMG = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf005" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf005" / "labels"
# Student output dataset yolo v11 conf 0.05
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf005" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf005" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf005" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf005" / "labels" / "val"
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.10
#PSEUDO_IMG = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf010" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf010" / "labels"
# Student output dataset yolo v11 conf 0.10
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf010" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf010" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf010" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf010" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.15
#PSEUDO_IMG = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf015" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf015" / "labels"
# Student output dataset yolo v11 conf 0.15
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf015" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf015" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf015" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf015" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.20
#PSEUDO_IMG = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf020" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf020" / "labels"
# Student output dataset yolo v11 conf 0.20
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf020" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf020" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf020" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf020" / "labels" / "val"
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.25
#PSEUDO_IMG = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf025" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf025" / "labels"
# Student output dataset yolo v11 conf 0.25
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf025" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf025" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf025" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf025" / "labels" / "val"
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.50
#PSEUDO_IMG = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf050" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf050" / "labels"
# Student output dataset yolo v11 conf 0.50
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf050" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf050" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf050" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf050" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.65
#PSEUDO_IMG = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf065" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11_v2"/"Yolov11_pseudo_conf065" / "labels"
# Student output dataset yolo v11 conf 0.65
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf065" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf065" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf065" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11_v2"/"student_yolo_v11_conf065" / "labels" / "val"
#---------------------------------------------------------------------------


#YOLO v11 Student labelling
#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.65)
#PSEUDO_IMG = SSL_ROOT / "Yolov11_pseudo_conf065" / "images"
#PSEUDO_LAB = SSL_ROOT / "Yolov11_pseudo_conf065" / "label"
# Student output dataset yolo v11 conf 0.65
#STUDENT_IMG_TRAIN = SSL_ROOT / "student_yolo_v11_conf065" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "student_yolo_v11_conf065" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "student_yolo_v11_conf065" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "student_yolo_v11_conf065" / "labels" / "val"
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.50)
#PSEUDO_IMG = SSL_ROOT / "Yolov11_pseudo_conf050" / "images"
#PSEUDO_LAB = SSL_ROOT / "Yolov11_pseudo_conf050" / "label"
# Student output dataset yolo v11 conf 0.50
#STUDENT_IMG_TRAIN = SSL_ROOT / "student_yolo_v11_conf050" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "student_yolo_v11_conf050" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "student_yolo_v11_conf050" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "student_yolo_v11_conf050" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.25)
#PSEUDO_IMG = SSL_ROOT / "Yolov11_pseudo_conf025" / "images"
#PSEUDO_LAB = SSL_ROOT / "Yolov11_pseudo_conf025" / "label"
# Student output dataset yolo v11 conf 0.25
#STUDENT_IMG_TRAIN = SSL_ROOT / "student_yolo_v11_conf025" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "student_yolo_v11_conf025" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "student_yolo_v11_conf025" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "student_yolo_v11_conf025" / "labels" / "val"
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.20)
#PSEUDO_IMG = SSL_ROOT / "YOLO_11"/"Yolov11_pseudo_conf020" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11"/"Yolov11_pseudo_conf020" / "labels"
# Student output dataset yolo v11 conf 0.20
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf020" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf020" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf020" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf020" / "labels" / "val"
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.15)
#PSEUDO_IMG = SSL_ROOT / "YOLO_11"/"Yolov11_pseudo_conf015" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11"/"Yolov11_pseudo_conf015" / "labels"
# Student output dataset yolo v11 conf 0.15
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf015" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf015" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf015" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf015" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.10)
#PSEUDO_IMG = SSL_ROOT / "YOLO_11"/"Yolov11_pseudo_conf010" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11"/"Yolov11_pseudo_conf010" / "labels"
# Student output dataset yolo v11 conf 0.10
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf010" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf010" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf010" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf010" / "labels" / "val"
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Pseudo-labeled data (conf 0.05)
#PSEUDO_IMG = SSL_ROOT / "YOLO_11"/"Yolov11_pseudo_conf005" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLO_11"/"Yolov11_pseudo_conf005" / "labels"
# Student output dataset yolo v11 conf 0.15
#STUDENT_IMG_TRAIN = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf005" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf005" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf005" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "YOLO_11"/"student_yolo_v11_conf005" / "labels" / "val"
#---------------------------------------------------------------------------

# Pseudo-labeled data (conf 0.25)
#PSEUDO_IMG = SSL_ROOT / "YOLOv8_Ab5_Pseudo" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLOv8_Ab5_Pseudo" / "label"
# Student output dataset yolo v8 conf 0.25 Ablation 05 Ratio
#STUDENT_IMG_TRAIN = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_05" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_05" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_05" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_05" / "labels" / "val"



#PSEUDO_IMG = SSL_ROOT / "YOLOv8_Ab10_Pseudo" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLOv8_Ab10_Pseudo" / "label"
# Student output dataset yolo v8 conf 0.25 Ablation 10 Ratio
#STUDENT_IMG_TRAIN = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_10" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_10" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_10" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_10" / "labels" / "val"


#PSEUDO_IMG = SSL_ROOT / "YOLOv8_Ab20_Pseudo" / "images"
#PSEUDO_LAB = SSL_ROOT / "YOLOv8_Ab20_Pseudo" / "label"
# Student output dataset yolo v8 conf 0.25 Ablation 10 Ratio
#STUDENT_IMG_TRAIN = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_20" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_20" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_20" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Ablation_SSL" /"V8_Student_AB_20" / "labels" / "val"
#============================================
#PSEUDO_IMG = SSL_AB_ROOT / "YOLOv8_Ab5_Pseudo" / "images"
#PSEUDO_LAB = SSL_AB_ROOT / "YOLOv8_Ab5_Pseudo" / "labels"
# Student output dataset yolo v8 conf 0.50 Ablation 05 Ratio
#STUDENT_IMG_TRAIN = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_5" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_5" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_5" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_5" / "labels" / "val"



#PSEUDO_IMG = SSL_AB_ROOT / "YOLOv8_Ab10_Pseudo" / "images"
#PSEUDO_LAB = SSL_AB_ROOT / "YOLOv8_Ab10_Pseudo" / "labels"
# Student output dataset yolo v8 conf 0.50 Ablation 10 Ratio
#STUDENT_IMG_TRAIN = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_10" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_10" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_10" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_10" / "labels" / "val"

#PSEUDO_IMG = SSL_AB_ROOT / "YOLOv8_Ab20_Pseudo" / "images"
#PSEUDO_LAB = SSL_AB_ROOT / "YOLOv8_Ab20_Pseudo" / "labels"
# Student output dataset yolo v8 conf 0.50 Ablation 20 Ratio
#STUDENT_IMG_TRAIN = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_20" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_20" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_20" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_20" / "labels" / "val"

#PSEUDO_IMG = SSL_AB_ROOT / "YOLOv8_Ab100_Pseudo" / "images"
#PSEUDO_LAB = SSL_AB_ROOT / "YOLOv8_Ab100_Pseudo" / "labels"
# Student output dataset yolo v8 conf 0.50 Ablation 20 Ratio
#STUDENT_IMG_TRAIN = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_100" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_100" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_100" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_AB_ROOT / "AB_STUDENT" /"V8_Student_AB_100" / "labels" / "val"


#==================== Optimized setting ========

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf005_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf005_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.05
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf005_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf005_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf005_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf005_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf010_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf010_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.10
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf010_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf010_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf010_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf010_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf015_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf015_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.15
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf015_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf015_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf015_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf015_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf020_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf020_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.20
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf020_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf020_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf020_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf020_Exp2" / "labels" / "val"


#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf025_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf025_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.25
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf025_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf025_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf025_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf025_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf050_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf050_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.50
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf050_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf050_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf050_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf050_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf065_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_08"/"Yolov08_pseudo_conf065_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.65
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf065_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf065_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf065_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_08"/"v8_student_pseudo_conf065_Exp2" / "labels" / "val"

#========== V11 ===============

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf005_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf005_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.05
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf005_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf005_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf005_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf005_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf010_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf010_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.10
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf010_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf010_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf010_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf010_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf015_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf015_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.15
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf015_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf015_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf015_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf015_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf020_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf020_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.20
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf020_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf020_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf020_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf020_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf025_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf025_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.25
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf025_Exp2" / "images" / "train"
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf025_Exp2" / "labels" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf025_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf025_Exp2" / "labels" / "val"

#PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf050_Exp2" / "images"
#PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf050_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.25
#STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf050_Exp2" / "labels" / "train"
#STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf050_Exp2" / "images" / "train"
#STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf050_Exp2" / "images" / "val"
#STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf050_Exp2" / "labels" / "val"

PSEUDO_IMG =SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf065_Exp2" / "images"
PSEUDO_LAB = SSL_ROOT / "Optimized"/"YOLO_V11"/"Yolov11_pseudo_conf065_Exp2" / "labels"
# Student output dataset yolo v8 conf 0.25
STUDENT_LAB_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf065_Exp2" / "labels" / "train"
STUDENT_IMG_TRAIN = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf065_Exp2" / "images" / "train"
STUDENT_IMG_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf065_Exp2" / "images" / "val"
STUDENT_LAB_VAL   = SSL_ROOT / "Optimized"/"YOLO_V11"/"v11_student_pseudo_conf065_Exp2" / "labels" / "val"

def copy_all(src_dir: Path, dst_dir: Path, pattern: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = list(src_dir.glob(pattern))
    for f in files:
        shutil.copy2(f, dst_dir / f.name)
    return len(files)

def main():
    # Create folders
    for d in [STUDENT_IMG_TRAIN, STUDENT_LAB_TRAIN, STUDENT_IMG_VAL, STUDENT_LAB_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) Copy labeled train
    n_img_lt = copy_all(LABELED_TRAIN_IMG, STUDENT_IMG_TRAIN, "*.*")
    n_lab_lt = copy_all(LABELED_TRAIN_LAB, STUDENT_LAB_TRAIN, "*.txt")

    # 2) Copy pseudo train (images + labels)
    n_img_p = copy_all(PSEUDO_IMG, STUDENT_IMG_TRAIN, "*.*")
    n_lab_p = copy_all(PSEUDO_LAB, STUDENT_LAB_TRAIN, "*.txt")

    # 3) Copy validation (keep original labeled val)
    n_img_v = copy_all(LABELED_VAL_IMG, STUDENT_IMG_VAL, "*.*")
    n_lab_v = copy_all(LABELED_VAL_LAB, STUDENT_LAB_VAL, "*.txt")

    print("✅ Student dataset created:")
    print(f"- labeled train images copied: {n_img_lt}")
    print(f"- labeled train labels copied: {n_lab_lt}")
    print(f"- pseudo images copied: {n_img_p}")
    print(f"- pseudo labels copied: {n_lab_p}")
    print(f"- val images copied: {n_img_v}")
    print(f"- val labels copied: {n_lab_v}")

    # Basic sanity checks: matching stems
    img_stems = {p.stem for p in STUDENT_IMG_TRAIN.glob('*.*')}
    lab_stems = {p.stem for p in STUDENT_LAB_TRAIN.glob('*.txt')}
    print("\nSanity check:")
    print("Train images:", len(img_stems))
    print("Train labels:", len(lab_stems))
    print("Images without labels:", len(img_stems - lab_stems))
    print("Labels without images:", len(lab_stems - img_stems))

if __name__ == "__main__":
    main()
