# Source code guide

This directory contains the **26 original Python scripts** supplied in `Source_FIle_29_sep_2026-20260929T073116Z-1-001.zip`. The scripts have been grouped by purpose; their filenames and code have not been changed. See [`docs/source-code-map.md`](../docs/source-code-map.md) for each file's location in the original ZIP.

The scripts document the research workflow, but the repository is **not yet an executable reproduction** of every journal experiment. The ZIP supplied scripts only. Dataset images, actual annotations, YAML configurations, videos, trained weights, and original training logs must be added or linked separately. Most scripts still contain paths from the original Windows computer.

## Workflow at a glance

| Stage | Folder | Task |
| --- | --- | --- |
| 1 | `data_preparation/` | Convert and check labels and dataset splits. |
| 2 | `training/` | Train a supervised YOLOv8 or YOLO11 teacher. |
| 3 | `pseudo_labeling/` | Predict labels for unlabeled images and assemble a student dataset. |
| 4 | `training/` | Train the SSL student. |
| 5 | `mean_teacher/` | Optionally repeat teacher predictions, student training, and EMA updates. |
| 6 | `evaluation/` | Compare training logs and evaluate annotated video frames or an MP4. |

The four paper configurations are indexed in [`configs/experiment_matrix.csv`](../configs/experiment_matrix.csv). One script's active settings do not cover every experiment or threshold.

## What each script does

### `data_preparation/`

| Script | Use |
| --- | --- |
| `convert_to_yolo.py` | Convert source `N` plus `x1 y1 x2 y2` labels to normalized YOLO boxes for train, validation, and test. Backs up and then replaces label files. |
| `covert_failed_to_yolo.py` | Repair failed conversions from the original backup files. Its filename is preserved as supplied. |
| `check_train_labels.py` | List counts and examples of train images without label files and labels without images. |
| `label_Check.py` | Check training label lines for YOLO's five fields and normalized coordinates. |
| `audit_dataset.py` | Audit labelled, test, and unlabeled folders. |
| `datasetInfo.py` | Summarize split counts, empty labels, infected images, boxes, and invalid labels. |
| `countImages.py` | Count images and files by dataset split. |
| `CreateEmptyFrame.py` | Make empty `.txt` labels for a chosen range of frame numbers; it **does not extract video frames**. |

An empty YOLO label file means no mite was annotated. Keep empty files when auditing healthy images and mite-free frames.

### `training/`

| Script | Use |
| --- | --- |
| `yolov8_train.py` | Train a supervised YOLOv8n model from a dataset YAML. |
| `train_yolov11_Supervised.py` | Train a supervised YOLO11n model from a dataset YAML. |
| `yolo8_ssl_student_train.py` | Start from a supervised YOLOv8 checkpoint and train with a student dataset YAML. |
| `train_yolov11_SSL_Conf025.py` | Start from a supervised YOLO11 checkpoint and train with a student dataset YAML. **Check the setting:** the filename says `025`, while its active YAML path and run name say `065`. |

All four supplied training scripts currently specify 50 epochs, batch size 8, image size 416, and `device=0`. Edit the selected script to represent a 320-pixel experiment.

### `pseudo_labeling/`

| Script | Use |
| --- | --- |
| `threshold_Sweep.py` | Count predictions and zero-detection images; it does **not** write pseudo-label files. Its active thresholds are `[0.25, 0.35, 0.50]`. |
| `Generate_Pseudolabel.py` | Predict on the unlabeled pool and write one YOLO `.txt` for every image, including empty files; copies the associated images. |
| `build_Student_dataset.py` | Copy human-labelled training images and one chosen pseudo-labelled set into a student dataset; retain human-labelled validation data. |
| `build_student_yolov11_conf025.py` | Alternative YOLO11 25% student dataset builder; deletes and recreates its configured output folder. |

`Generate_Pseudolabel.py` and `build_Student_dataset.py` have many commented paths for earlier experiments. Only their uncommented assignments run. In this ZIP, the active paths point to an optimized YOLO11 65% run.

### `mean_teacher/`

| Script | Use |
| --- | --- |
| `mean_teacher.py` | Two-round configured loop: teacher pseudo labels → student dataset → student training → EMA teacher update. |
| `ema_Update_Round1.py` | Standalone EMA update from existing round-0 teacher and round-1 student weights. |
| `mean_teacher_r2.py` | Separate YOLOv8 round-2 workflow using an existing round-1 student checkpoint. |

`mean_teacher.py` currently uses 416 pixels, 25% pseudo labels, EMA decay 0.99, and **20 student epochs per round**. `mean_teacher_r2.py` uses 640 pixels and 20 epochs. The standalone scripts are an alternative manual workflow; the complete loop already updates its teacher after each round.

### `evaluation/`

| Script | Use |
| --- | --- |
| `day5_baseline_extract.py` | Read one Ultralytics `results.csv`, select the best epoch, and save a baseline row. Its active metadata says 20 epochs. |
| `day5_Build_comparison.py` | Read configured `results.csv` files and write comparison CSV and Markdown tables. It reassigns `EXPERIMENTS` several times; only the **last active list** runs. |
| `validate_video.py` | Compute metrics using a checkpoint and an annotated video-frame dataset YAML; also save example predictions. |
| `VideoValidation.py` | Predict on an MP4 and save an annotated MP4 and optionally a detection CSV. This is visual inference, **not** a ground-truth mAP calculation. |
| `predictVideo.py` | Minimal example for predicting on one video. |

### `experiments/` and `tools/`

| Script | Use |
| --- | --- |
| `experiments/ablation_study.py` | Create stratified 5%, 10%, 20%, and 100% human-labelled training subsets, plus YAML files. This optional ablation differs from the paper's four main configurations. |
| `tools/version.py` | Report local PyTorch CUDA and GPU availability. |

## Before running on your computer

1. **Find the original inputs.** You need human-labelled train/validation/test images and YOLO labels, the unlabeled pool, dataset YAMLs, the checkpoints you intend to reuse, original `results.csv` logs, and annotated video frames. The repository's CSV templates are not substitutes for those files.
2. **Use a copy of the dataset.** The converter overwrites labels after backing them up. The repair script can replace labels. `CreateEmptyFrame.py` creates or truncates frame label files. Student builders copy many files; one deletes its output directory before rebuilding.
3. **Configure one run at a time.** Replace the active `C:\Users\...` paths in the relevant scripts with real paths. Align the model family, human-labelled split, unlabeled split, teacher checkpoint, confidence threshold, image size, YAML, and output name. Check that validation and test images never enter the student training set.
4. **Prepare Python dependencies.** The code imports `ultralytics`, `torch`, `PIL` (Pillow), `pandas`, and `cv2` (OpenCV). The comparison script's Markdown export needs the dependency used by `pandas.to_markdown`. Record the versions used in the original research environment; no verified dependency lock file came with this ZIP.
5. **Check the compute device.** `device=0` assumes a compatible GPU. If your Mac has no compatible device configured, change the selected script's `device` or `DEVICE` to `"cpu"` for a small trial. A different device may affect speed and exact reproduction.

Large images and videos belong in the ignored `data/raw/`, `data/frames/`, or `data/videos/` paths, or in an external dataset location. Actual small YOLO `.txt` labels may be committed under `data/labels/` when permissions allow. The legacy scripts will not find those repository paths until their constants are changed. Put checkpoint identifiers, links, and checksums in `models/checkpoints.csv`; `.pt` files and `runs/` output are ignored by Git.

## Run order for one experiment

These commands identify entry points. **They will not work until you have the inputs and have edited the active settings above.** Run them from the repository root in your prepared Python environment.

1. **Prepare and check labels.** If the source labels are still in `N`/coordinate format, run the converter **on a copy**; use the repair script only when backups exist and conversion failed. Then inspect the splits:

   ```bash
   python src/data_preparation/convert_to_yolo.py
   python src/data_preparation/check_train_labels.py
   python src/data_preparation/label_Check.py
   python src/data_preparation/datasetInfo.py
   python src/data_preparation/audit_dataset.py
   ```

2. **Train one supervised teacher.** Set `data` to the YAML for that experiment's human-labelled train and validation data. Set `imgsz` to 320 or 416 and choose one model:

   ```bash
   python src/training/yolov8_train.py
   # or
   python src/training/train_yolov11_Supervised.py
   ```

3. **Generate pseudo labels.** Set `TEACHER_WEIGHTS`, `UNLABELED_IMG_DIR`, `PSEUDO_LABEL_DIR`, `PSEUDO_IMG_DIR`, `CONF_THRES`, and `IMGSZ` in the generator. Give each model/experiment/threshold a separate output folder:

   ```bash
   python src/pseudo_labeling/Generate_Pseudolabel.py
   ```

   Inspect box counts and empty files. The optional count-only helper's defaults are **not** the paper's seven thresholds:

   ```bash
   python src/pseudo_labeling/threshold_Sweep.py
   ```

4. **Build and train one student.** Set the builder's human and pseudo input paths for the same experiment. Ensure the validation set uses human labels. Point the training YAML to the resulting dataset. Choose the correct model script:

   ```bash
   python src/pseudo_labeling/build_Student_dataset.py
   python src/training/yolo8_ssl_student_train.py
   # For YOLO11, configure a matching student dataset, then use:
   python src/training/train_yolov11_SSL_Conf025.py
   ```

   The YOLO11 training file's active run is `065`; verify it before treating it as a 25% run.

5. **Run Mean Teacher if applicable.** After setting the initial supervised checkpoint and all data paths, `mean_teacher.py` performs both rounds. Use `ema_Update_Round1.py` and `mean_teacher_r2.py` only for a separate manual YOLOv8 reconstruction from verified checkpoints:

   ```bash
   python src/mean_teacher/mean_teacher.py
   ```

6. **Evaluate saved checkpoints.** Compare original run CSVs with the table scripts. Use `validate_video.py` with an annotated frame dataset YAML for quantitative video metrics; use `VideoValidation.py` with an MP4 for a visual preview:

   ```bash
   python src/evaluation/validate_video.py
   python src/evaluation/VideoValidation.py
   ```

   Frame extraction and the complete 600-frame annotation process are **not provided** in this source ZIP. Prediction CSVs alone cannot replace human ground truth.

## What to record for each published result

Record the paper experiment ID, YOLO version, IDs of the human-labelled and unlabeled images, dataset YAML, teacher checkpoint, confidence threshold, number of pseudo boxes and empty labels, student checkpoint, image size, epochs, batch size, random seed, software versions, original `results.csv`, and validation/test dataset. Use `configs/`, `data/manifests/`, `data/splits/`, `models/checkpoints.csv`, and `results/` for these records.

The journal reports thresholds `0.05`, `0.10`, `0.15`, `0.20`, `0.25`, `0.50`, and `0.65`. The uploaded `threshold_Sweep.py` lists three different values; `Generate_Pseudolabel.py` has commented selections rather than an automatic full sweep. The scripts supplied here also omit the YAMLs, checkpoint weights, labels, raw run CSVs, videos, and frame extraction script. Check [`docs/paper-mapping.md`](../docs/paper-mapping.md) and the original outputs before linking a script to a journal table. Keep the manuscript PDF as a reference; **do not use its pages or figures as training images or annotations**.
