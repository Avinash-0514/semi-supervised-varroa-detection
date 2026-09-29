from pathlib import Path

root = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Labelled")
img_dir = root / "images" / "train"
lab_dir = root / "labels" / "train"

imgs = {p.stem for p in img_dir.glob("*.*")}
labs = {p.stem for p in lab_dir.glob("*.txt")}

print("Train images:", len(imgs))
print("Train labels:", len(labs))
print("Images without labels:", len(imgs - labs))
print("Labels without images:", len(labs - imgs))

missing = list(imgs - labs)[:5]
extra = list(labs - imgs)[:5]

if missing:
    print("Example missing labels:", missing)
if extra:
    print("Example extra labels:", extra)

