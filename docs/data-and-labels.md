# Data and label rules

The target object is a **Varroa mite bounding box**, not a class for the whole bee. Source images also include healthy bees with no visible mite. The raw source dataset contains 13,509 images; this repository must document which images from it formed each research subset.

- `data/splits/train_915.csv`: original labelled training group used in Experiments 1 and 3.
- `data/splits/train_additional_406.csv`: additional human-labelled group used with the 915 in Experiments 2 and 4.
- `data/splits/unlabeled_1281.csv`: pseudo-labelling candidate pool. It must stay separate from the held-out evaluation images.
- `data/splits/validation.csv` and `test.csv`: fixed evaluation IDs. Populate the actual IDs used; do not fill them from aggregate dataset statistics alone.

The manuscript describes normalized YOLO box geometry `(x_center, y_center, width, height)` for mite detections. Record the exact class-ID mapping from your original training data before publishing it. Include an empty label or explicit zero-box indicator when an image has no visible mite; a missing file must not silently mean 'healthy'.

Keep the origin of each box clear: source dataset/manual addition, machine-generated pseudo-label, or manual video-frame label. For each pseudo-label set, record the experiment, detector, seed/checkpoint, confidence threshold, image ID, accepted boxes with confidence scores, and every selected image that yielded zero boxes. Human test/video labels must never be mixed into pseudo-label training.

CSV files in this template contain **headers only** until original data are supplied. Never create annotation rows by copying plotted values or guessing image IDs.
