# Paper to repository mapping

This mapping follows the supplied journal draft dated 26 September 2026. The values below are paper claims, not independently reproduced results.

| Paper location | Required experiment material |
| --- | --- |
| Section 3.8, Table 2 | Source version, 915 + 406 image IDs, 1281 unlabelled IDs, common evaluation split, 320/416 image size |
| Table 3 | Baseline and best SSL runs for both architectures in Experiments 1-4 (8 baselines, 8 best SSL results) |
| Figures 2 and 3 | Metrics for all seven thresholds in all 8 experiment/architecture combinations (up to 56 SSL runs) |
| Table 4, Figure 4 | Threshold choice and best result for each of the 8 combinations |
| Table 5, Figure 5 | Mean Teacher 25% results for 4 experiments x 2 architectures; retain both reported rounds, not only the best |
| Section 3.10, Table 6, Figure 6 | Video source/frame manifest, 600 manually annotated frames, exact four model/checkpoint IDs, Easy and Medium metrics |
| Section 4.6 | Hard-condition sample frames/predictions and qualitative assessment; the draft does not report directly comparable quantitative metrics for Hard |

### Practical upload priority

1. Experiment 4, YOLOv8 at 25% is the highest image-domain SSL result (mAP@0.5 = 0.704), but it cannot substantiate the paper's comparisons by itself.
2. Add the baselines and all four experiment settings, both detector families, and seven thresholds to support the main comparisons.
3. Add the Mean Teacher rounds and video validation artifacts to support the remaining results.

The video caption says *optimised SSL 25%*. Experiment 4 YOLOv11's image-domain optimum is *10%*. Consequently the best static-image YOLOv11 checkpoint must not automatically be labelled as the video checkpoint. Resolve the original run IDs before filling `models/checkpoints.csv`.
