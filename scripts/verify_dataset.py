from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Processed dataset folder
root = ROOT / "Processed_Dataset"

print("Dataset Path:", root)
print()

if not root.exists():
    raise FileNotFoundError(f"{root} does not exist")

patients = [p for p in root.iterdir() if p.is_dir()]

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

print("Patients      :", len(patients))

t1 = 0
lesion = 0

for p in patients:

    if (p / "T1.nii.gz").exists():
        t1 += 1

    elif (p / "T1.nii").exists():
        t1 += 1

    if (p / "Lesion.nii.gz").exists():
        lesion += 1

    elif (p / "Lesion.nii").exists():
        lesion += 1

print("T1 Files      :", t1)
print("Lesion Files  :", lesion)