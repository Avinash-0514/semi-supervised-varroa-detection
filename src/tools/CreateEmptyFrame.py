import os

# 👉 OUTPUT folder
output_folder = r"C:\Users\User\Desktop\Thesis\Src_Yolo\labelImg\video2_medium"
os.makedirs(output_folder, exist_ok=True)

# 👉 Naming
base_name = "frame"

# 👉 Range
start = 37
end = 150   # inclusive

# 👉 Create files
for i in range(start, end + 1):
    file_name = f"{base_name}_{i:04d}.txt"
    file_path = os.path.join(output_folder, file_name)

    with open(file_path, "w") as f:
        pass  # empty file

print("✅ TXT files created successfully!")