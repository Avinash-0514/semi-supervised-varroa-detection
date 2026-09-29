from pathlib import Path

LABEL_DIR = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo\SSL\Labelled\labels\train")


bad_files = []

for txt in LABEL_DIR.glob("*.txt"):
    with open(txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                bad_files.append(txt.name)
                break
            try:
                nums = list(map(float, parts[1:]))
                if any(v < 0 or v > 1 for v in nums):
                    bad_files.append(txt.name)
                    break
            except:
                bad_files.append(txt.name)
                break

if bad_files:
    print("❌ Non-YOLO labels found:")
    for f in bad_files[:10]:
        print(f)
else:
    print("✅ All labels are valid YOLO format")