# Before making a journal release

- [ ] Compare each CSV of paper claims with raw run logs; record the run IDs and exact train/validation/test image IDs.
- [ ] Include all four experiment settings, both architectures, all threshold outcomes, and Mean Teacher rounds or mark anything unavailable.
- [ ] Verify each manual/pseudo/video label against its source image/frame, including empty cases.
- [ ] Check shared evaluation splits and no training leakage from validation/test/video.
- [ ] Record original data citation, versions, rights, and permissions for every public video or redistributed annotation.
- [ ] Add environment/dependency versions, seeds, commands, trained-checkpoint provenance and checksums.
- [ ] Verify Figure 6 and Table 6 checkpoint naming and the 25% vs 10% YOLOv11 threshold distinction.
- [ ] Replace manuscript placeholders with final publication details; choose an explicit licence for your own code and annotations.
- [ ] Tag the checked paper version and, if desired, archive that public release for a DOI.

The supplied PDF currently contains draft journal/DOI placeholders. Do not cite those placeholders as a published identifier.
