# Video validation material

Section 3.10 states that 600 frames were manually annotated. Table 6 reports Easy and Medium scores for four selected optimised SSL 25% models; the Hard condition is interpreted qualitatively.

Fill `data/manifests/video_clips.csv` with source, permission or licence, checksums, original fps, and scenario. Fill `data/manifests/video_frames.csv` with the exact source clip and frame index or timestamp for each extracted frame. Put manually drawn mite boxes (and zero-box frames) into `data/labels/video/video_boxes.csv`. Put the exact checkpoint links and checksums into `models/checkpoints.csv`.

The draft states 30 fps and roughly 150 frames per 5-second segment. Do not infer a count per scenario from this statement: use the actual source manifest. If the videos cannot be shared, document access restrictions honestly; labels alone cannot enable independent recomputation of video metrics without the frames.

Before release, verify the paper's 915/1321 video model descriptions and the Figure 6 Exp.1/Exp.2 labels against the real files. The draft's caption mentions SSL 25% for *all four* video models, including YOLOv11 with 1321 labels, whereas that model's best static-image threshold in Table 3 is 10%.
