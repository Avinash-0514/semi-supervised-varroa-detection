import os

# ======= CONFIGURE YOUR DATASET ROOT HERE =======
DATASET_ROOT = "VarroaDataset"  # change if needed

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def count_files(folder, extensions=None):
    if not os.path.exists(folder):
        return 0
    
    count = 0
    for file in os.listdir(folder):
        if extensions:
            if file.lower().endswith(extensions):
                count += 1
        else:
            count += 1
    return count


def check_split(split_name):
    image_folder = os.path.join(DATASET_ROOT, "images", split_name)
    label_folder = os.path.join(DATASET_ROOT, "labels", split_name)

    image_count = count_files(image_folder, IMAGE_EXTENSIONS)
    label_count = count_files(label_folder, (".txt",))

    print(f"\n=== {split_name.upper()} ===")
    print(f"Images: {image_count}")
    print(f"Label files: {label_count}")
    print(f"Difference (Images - Labels): {image_count - label_count}")


def main():
    print("=== DATASET COUNT SUMMARY ===")

    splits = [
        "train_labeled",
        "train_unlabeled",
        "train",
        "val",
        "test"
    ]

    for split in splits:
        check_split(split)


if __name__ == "__main__":
    main()