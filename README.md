# Varroa mite detection with semi-supervised learning

Repository template for the experiments described in *Reducing Annotation Dependency in Tiny Object Detection through Semi-Supervised Learning: Varroa Mite Detection as a Case Study* (journal draft, 26 September 2026).

**Status:** This is a structure and an index of paper-reported numbers. It does not yet contain the original research code, images, actual labels, videos, trained weights, or verified run logs. Do not claim reproducibility until the real files are added and checked. The manuscript PDF is intentionally not packaged here.

## What the paper compares

| Experiment | Labelled training images | Input size | Purpose |
| --- | ---: | ---: | --- |
| 1 | 915 | 320 x 320 | Label-scarce reference |
| 2 | 1321 (915 + 406) | 320 x 320 | More human labels |
| 3 | 915 | 416 x 416 | Higher input resolution |
| 4 | 1321 (915 + 406) | 416 x 416 | Both changes |

Each row is evaluated with YOLOv8 and YOLOv11. The paper compares supervised baselines with pseudo-labelling at confidence thresholds 5%, 10%, 15%, 20%, 25%, 50%, and 65%. Mean Teacher is reported at 25% for two rounds per configuration. The video evaluation uses four selected 25% SSL models across Easy and Medium conditions; Hard is qualitative. See `docs/paper-mapping.md` for exact claims.

## Where your real files belong

- `src/`: original data preparation, baseline training, pseudo-labelling, Mean Teacher, evaluation, and video scripts or notebooks.
- `configs/`: the eight paper configurations and threshold list. Add exact pretrained weights, package versions, seeds, augmentation, and Mean Teacher EMA details when verified.
- `data/splits/`: image IDs for the 915 labelled images, 406 additional labelled images, 1281 unlabelled images, and fixed validation/test sets. The CSV files are empty templates.
- `data/labels/human/`: actual mite bounding-box annotations for the static-image training and evaluation subsets, with provenance.
- `data/labels/pseudo/`: the final generated labels used by each threshold/model/experiment; include accepted and zero-label image IDs.
- `data/labels/video/`: the actual human annotations for video-derived frames, including frames with no visible mite.
- `data/manifests/`: image and video/frame IDs to connect every annotation to its source.
- `results/`: paper-reported values and, later, verified individual run logs.
- `models/`: checkpoint links and checksums, rather than accidentally committing many large model files.

## Data and storage

The source VarroaDataset is available at https://doi.org/10.5281/zenodo.4085044 . Record the exact source version and file checksums. Reference the original images by stable image ID instead of copying the whole original dataset into Git. Store large shareable videos and checkpoints outside Git, with a stable download link and checksum here. Confirm redistribution rights before releasing third-party images, videos, or annotations.

Keep the labels in Git when practical. If there are too many small pseudo-label files, a CSV with one row per accepted box plus a manifest of all selected images (including zero-label images) is easier to audit and can be converted back to YOLO TXT format using your original conversion code.

## How to populate this template

1. Fill the split lists from the files actually used during the thesis. Check for train/validation/test leakage and identify whether the extra 406 labels are independent of the 915.
2. Copy the original code and dependency/environment information. Record the exact checkpoint used to generate each pseudo-label set.
3. Add the true human and generated labels. Do not substitute labels extracted from the manuscript figures.
4. Enter complete threshold and two-round Mean Teacher metrics from original experiment logs. Recreate Figures 2-6 and Tables 3, 5, and 6.
5. Match the four video models to their exact weights. Table 6 describes them as optimised 25% SSL models, while Figure 6 uses Exp.1/Exp.2 labels; do not guess the mapping.
6. Verify the files and their publication permissions before publishing a release.

To start your own GitHub repository after unpacking this template:

```bash
git init
git add .
git commit -m "Add journal experiment repository structure"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the example remote with your own repository URL. The template can remain private while you assemble and verify the real data.
