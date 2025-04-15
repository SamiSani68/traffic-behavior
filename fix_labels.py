import os

LABEL_DIR = "/home/fullsuper/Sami/dataset/labels/test"
INVALID_CLASSES = {"7", "8"}

for root, _, files in os.walk(LABEL_DIR):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                lines = f.readlines()

            valid_lines = [line for line in lines if line.strip().split()[0] not in INVALID_CLASSES]

            with open(path, "w") as f:
                f.writelines(valid_lines)
