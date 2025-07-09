import numpy as np
from pathlib import Path

# === Directories ===
NPY_DIR = Path("behavior/homography_matrices_multi")
CSV_DIR = Path("behavior/homography_csv_multi")
CSV_DIR.mkdir(parents=True, exist_ok=True)

for npy_file in sorted(NPY_DIR.glob("*.npy")):
    try:
        H = np.load(npy_file)
        csv_file = CSV_DIR / (npy_file.stem + ".csv")
        np.savetxt(csv_file, H, delimiter=",", fmt="%.6f")
        print(f"✅ Converted: {npy_file.name} → {csv_file.name}")
    except Exception as e:
        print(f"Failed to convert {npy_file.name}: {e}")

print("All homography matrices converted to CSV.")
print(f"Output folder: {CSV_DIR}")
