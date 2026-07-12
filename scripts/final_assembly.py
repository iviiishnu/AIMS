"""
============================================================
AIMS-TBI 2026
Final Dataset Assembly Pipeline

Author : Vishnu Vardhan
Purpose:
    Build the final training-ready dataset from the
    official challenge files.

Pipeline

1. Environment Check
2. Dataset Verification
3. Metadata Loading
4. Image Processing
5. Dataset Generation
6. Quality Control
7. Final ZIP

============================================================
"""

# ============================================================
# Imports
# ============================================================

import os
import json
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm

# ============================================================
# Project Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "ChallengeFiles"

METADATA_DIR = ROOT / "metadata"

OUTPUT_DIR = ROOT / "Processed_Dataset"

LOG_DIR = ROOT / "logs"

OUTPUT_DIR.mkdir(exist_ok=True)

LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# Target Standard
# ============================================================

TARGET_ORIENTATION = "LAS"

TARGET_SPACING = (
    1.0,
    1.0,
    1.0
)

# ============================================================
# Metadata Files
# ============================================================

MANIFEST_FILE = METADATA_DIR / "dataset_manifest.csv"

ORIENTATION_FILE = METADATA_DIR / "reorientation_log.csv"

RESAMPLING_FILE = METADATA_DIR / "resampling_report.csv"

# ============================================================
# Startup
# ============================================================

print("=" * 70)
print("AIMS-TBI FINAL DATASET ASSEMBLY")
print("=" * 70)

print()

print("Project Root")
print(ROOT)

print()

print("Dataset")
print(DATASET_DIR)

print()

print("Metadata")
print(METADATA_DIR)

print()

print("Output")
print(OUTPUT_DIR)

print()

print("Logs")
print(LOG_DIR)

print()

# ============================================================
# Folder Verification
# ============================================================

required_dirs = [
    DATASET_DIR,
    METADATA_DIR
]

missing = []

for folder in required_dirs:

    if not folder.exists():

        missing.append(folder)

if len(missing) > 0:

    print("Missing directories:")

    for folder in missing:

        print(folder)

    raise SystemExit()

print("Folder verification passed.")

print()

# ============================================================
# Metadata Verification
# ============================================================

required_files = [
    MANIFEST_FILE,
    ORIENTATION_FILE,
    RESAMPLING_FILE
]

missing = []

for file in required_files:

    if not file.exists():

        missing.append(file)

if len(missing) > 0:

    print("Missing metadata files:")

    for file in missing:

        print(file)

    raise SystemExit()

print("Metadata verification passed.")

print()

# ============================================================
# Load Metadata
# ============================================================

manifest = pd.read_csv(MANIFEST_FILE)

orientation_log = pd.read_csv(ORIENTATION_FILE)

resampling_log = pd.read_csv(RESAMPLING_FILE)

print("Patients")

print(len(manifest))

print()

print("Orientation Corrections")

print(
    orientation_log["PatientID"].nunique()
)

print()

print("Resampling Corrections")

print(
    resampling_log["PatientID"].nunique()
)

print()

print("=" * 70)
print("MODULE 1 COMPLETED")
print("=" * 70)
# ============================================================
# MODULE 2
# File Discovery & Dataset Verification
# ============================================================

print()
print("=" * 70)
print("MODULE 2 - FILE DISCOVERY")
print("=" * 70)


# ------------------------------------------------------------
# Locate image (.nii or .nii.gz)
# ------------------------------------------------------------

def find_image(patient_id, modality):
    """
    Returns the full path of the requested image.

    Supports both:
        scan_xxxx_T1.nii
        scan_xxxx_T1.nii.gz
    """

    candidates = [

        DATASET_DIR / f"{patient_id}_{modality}.nii",

        DATASET_DIR / f"{patient_id}_{modality}.nii.gz"

    ]

    for file in candidates:

        if file.exists():

            return file

    return None


# ------------------------------------------------------------
# Verify Complete Dataset
# ------------------------------------------------------------

missing = []

verified = 0

for patient in manifest["PatientID"]:

    t1 = find_image(patient, "T1")

    lesion = find_image(patient, "Lesion")

    if t1 is None:

        missing.append(
            f"{patient}_T1"
        )

    if lesion is None:

        missing.append(
            f"{patient}_Lesion"
        )

    if t1 is not None and lesion is not None:

        verified += 1


print()

print("Verified Patients :", verified)

print()

print("Missing Files :", len(missing))

if len(missing) > 0:

    print()

    print("First Missing")

    for item in missing[:10]:

        print(item)

else:

    print()

    print("✓ Dataset verification passed.")

print()

print("=" * 70)
print("MODULE 2 COMPLETED")
print("=" * 70)
# ============================================================
# MODULE 3
# Core Helper Functions
# ============================================================

print()
print("=" * 70)
print("MODULE 3 - HELPER FUNCTIONS")
print("=" * 70)

# ------------------------------------------------------------
# Orientation
# ------------------------------------------------------------

def get_orientation(image):

    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )


def reorient_image(image, target_orientation=TARGET_ORIENTATION):

    current = get_orientation(image)

    if current == target_orientation:

        return image

    orienter = sitk.DICOMOrientImageFilter()

    orienter.SetDesiredCoordinateOrientation(
        target_orientation
    )

    return orienter.Execute(image)


# ------------------------------------------------------------
# Resampling
# ------------------------------------------------------------

def resample_image(image, target_spacing, is_label=False):

    original_spacing = image.GetSpacing()

    original_size = image.GetSize()

    new_size = [

        int(
            round(
                original_size[i]
                * original_spacing[i]
                / target_spacing[i]
            )
        )

        for i in range(3)

    ]

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(target_spacing)

    resampler.SetSize(new_size)

    resampler.SetOutputOrigin(
        image.GetOrigin()
    )

    resampler.SetOutputDirection(
        image.GetDirection()
    )

    resampler.SetTransform(
        sitk.Transform()
    )

    resampler.SetDefaultPixelValue(0)

    if is_label:

        resampler.SetInterpolator(
            sitk.sitkNearestNeighbor
        )

    else:

        resampler.SetInterpolator(
            sitk.sitkLinear
        )

    return resampler.Execute(image)


# ------------------------------------------------------------
# Read / Write
# ------------------------------------------------------------

def load_patient(patient_id):

    t1 = sitk.ReadImage(
        str(find_image(patient_id, "T1"))
    )

    lesion = sitk.ReadImage(
        str(find_image(patient_id, "Lesion"))
    )

    return t1, lesion


def save_patient(patient_id, t1, lesion):

    patient_dir = OUTPUT_DIR / patient_id

    patient_dir.mkdir(
        exist_ok=True
    )

    sitk.WriteImage(
        t1,
        str(patient_dir / "T1.nii.gz")
    )

    sitk.WriteImage(
        lesion,
        str(patient_dir / "Lesion.nii.gz")
    )


print()

print("Orientation Helper      ✓")

print("Resampling Helper       ✓")

print("Load/Save Helpers       ✓")

print()

print("=" * 70)

print("MODULE 3 COMPLETED")

print("=" * 70)
# ============================================================
# MODULE 4
# Process One Patient
# ============================================================

print()
print("=" * 70)
print("MODULE 4 - PROCESS SINGLE PATIENT")
print("=" * 70)

# ------------------------------------------------------------
# Test Patient
# ------------------------------------------------------------

TEST_PATIENT = "scan_0053"

print()
print("Patient :", TEST_PATIENT)

# ------------------------------------------------------------
# Load Images
# ------------------------------------------------------------

t1, lesion = load_patient(TEST_PATIENT)

print()

print("Original Orientation :", get_orientation(t1))
print("Original Spacing     :", t1.GetSpacing())
print("Original Size        :", t1.GetSize())

# ------------------------------------------------------------
# Check Metadata
# ------------------------------------------------------------

needs_reorientation = (
    TEST_PATIENT in
    set(orientation_log["PatientID"])
)

needs_resampling = (
    TEST_PATIENT in
    set(resampling_log["PatientID"])
)

print()

print("Needs Reorientation :", needs_reorientation)
print("Needs Resampling    :", needs_resampling)

# ------------------------------------------------------------
# Apply Reorientation
# ------------------------------------------------------------

if needs_reorientation:

    t1 = reorient_image(t1)

    lesion = reorient_image(lesion)

# ------------------------------------------------------------
# Apply Resampling
# ------------------------------------------------------------

if needs_resampling:

    t1 = resample_image(
        t1,
        TARGET_SPACING,
        is_label=False
    )

    lesion = resample_image(
        lesion,
        TARGET_SPACING,
        is_label=True
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

save_patient(
    TEST_PATIENT,
    t1,
    lesion
)

print()

print("Saved Successfully")

print(OUTPUT_DIR / TEST_PATIENT)

# ------------------------------------------------------------
# Verify Saved Images
# ------------------------------------------------------------

saved_t1 = sitk.ReadImage(
    str(
        OUTPUT_DIR /
        TEST_PATIENT /
        "T1.nii.gz"
    )
)

saved_lesion = sitk.ReadImage(
    str(
        OUTPUT_DIR /
        TEST_PATIENT /
        "Lesion.nii.gz"
    )
)

print()

print("Processed Orientation :", get_orientation(saved_t1))
print("Processed Spacing     :", saved_t1.GetSpacing())
print("Processed Size        :", saved_t1.GetSize())

print()

print("Lesion Orientation    :", get_orientation(saved_lesion))
print("Lesion Spacing        :", saved_lesion.GetSpacing())

print()

print("=" * 70)
print("MODULE 4 COMPLETED")
print("=" * 70)
# ============================================================
# MODULE 5
# Process Entire Dataset
# ============================================================

print()
print("=" * 70)
print("MODULE 5 - PROCESS ENTIRE DATASET")
print("=" * 70)

processing_log = []

success = 0
failed = 0

for patient in tqdm(manifest["PatientID"], desc="Processing"):

    try:

        # -----------------------------
        # Load
        # -----------------------------

        t1, lesion = load_patient(patient)

        # -----------------------------
        # Reorientation
        # -----------------------------

        if patient in set(orientation_log["PatientID"]):

            t1 = reorient_image(t1)

            lesion = reorient_image(lesion)

        # -----------------------------
        # Resampling
        # -----------------------------

        if patient in set(resampling_log["PatientID"]):

            t1 = resample_image(
                t1,
                TARGET_SPACING,
                is_label=False
            )

            lesion = resample_image(
                lesion,
                TARGET_SPACING,
                is_label=True
            )

        # -----------------------------
        # Save
        # -----------------------------

        save_patient(
            patient,
            t1,
            lesion
        )

        processing_log.append({
            "PatientID": patient,
            "Status": "Success"
        })

        success += 1

    except Exception as e:

        processing_log.append({
            "PatientID": patient,
            "Status": "Failed",
            "Reason": str(e)
        })

        failed += 1

print()

print("=" * 70)
print("PROCESSING FINISHED")
print("=" * 70)

print()

print("Successful :", success)

print("Failed     :", failed)

print()

processing_log = pd.DataFrame(processing_log)

processing_log.to_csv(
    LOG_DIR / "processing_log.csv",
    index=False
)

print("Processing log saved.")
# ============================================================
# MODULE 6
# Generate dataset.csv
# ============================================================

print()
print("=" * 70)
print("MODULE 6 - GENERATING dataset.csv")
print("=" * 70)

records = []

patients = sorted([
    p.name
    for p in OUTPUT_DIR.iterdir()
    if p.is_dir()
])

for patient in patients:

    patient_dir = OUTPUT_DIR / patient

    # --------------------------------------------------------
    # Find T1
    # --------------------------------------------------------

    if (patient_dir / "T1.nii.gz").exists():

        t1 = patient_dir / "T1.nii.gz"

    else:

        t1 = patient_dir / "T1.nii"

    # --------------------------------------------------------
    # Find Lesion
    # --------------------------------------------------------

    if (patient_dir / "Lesion.nii.gz").exists():

        lesion = patient_dir / "Lesion.nii.gz"

    else:

        lesion = patient_dir / "Lesion.nii"

    records.append({

        "PatientID": patient,

        "T1": str(t1.relative_to(ROOT)),

        "Lesion": str(lesion.relative_to(ROOT))

    })

dataset = pd.DataFrame(records)

dataset.sort_values("PatientID", inplace=True)

dataset.to_csv(

    OUTPUT_DIR / "dataset.csv",

    index=False

)

print()

print("dataset.csv created successfully")

print()

print("Patients :", len(dataset))

print()

print(dataset.head())

print()

print("=" * 70)
print("MODULE 6 COMPLETED")
print("=" * 70)
# ============================================================
# MODULE 7
# Generate dataset.json (MONAI Compatible)
# ============================================================

print()
print("=" * 70)
print("MODULE 7 - GENERATING dataset.json")
print("=" * 70)

dataset_json = {
    "name": "AIMS-TBI-2026",
    "description": "Preprocessed T1 + Lesion dataset for lesion segmentation",
    "tensorImageSize": "3D",
    "modality": {
        "0": "T1",
        "1": "dMRI"
    },
    "labels": {
        "0": "Background",
        "1": "Lesion"
    },
    "numTraining": len(dataset),
    "training": []
}

for _, row in dataset.iterrows():

    patient = row["PatientID"]

    patient_dir = OUTPUT_DIR / patient

    # -------------------------
    # T1
    # -------------------------

    if (patient_dir / "T1.nii.gz").exists():
        t1 = str((patient_dir / "T1.nii.gz").relative_to(ROOT))
    else:
        t1 = str((patient_dir / "T1.nii").relative_to(ROOT))

    # -------------------------
    # Lesion
    # -------------------------

    if (patient_dir / "Lesion.nii.gz").exists():
        lesion = str((patient_dir / "Lesion.nii.gz").relative_to(ROOT))
    else:
        lesion = str((patient_dir / "Lesion.nii").relative_to(ROOT))

    # -------------------------
    # dMRI (future support)
    # -------------------------

    dmri = None

    if (patient_dir / "dMRI.nii.gz").exists():
        dmri = str((patient_dir / "dMRI.nii.gz").relative_to(ROOT))

    elif (patient_dir / "dMRI.nii").exists():
        dmri = str((patient_dir / "dMRI.nii").relative_to(ROOT))

    dataset_json["training"].append({

        "patient_id": patient,

        "image": t1,

        "label": lesion,

        "dmri": dmri

    })

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

json_path = OUTPUT_DIR / "dataset.json"

with open(json_path, "w") as f:

    json.dump(
        dataset_json,
        f,
        indent=4
    )

print()

print("dataset.json created successfully")

print()

print("Training Samples :", len(dataset_json["training"]))

print()

print("Saved to")

print(json_path)

print()

print("=" * 70)
print("MODULE 7 COMPLETED")
print("=" * 70)
# ============================================================
# MODULE 8
# Final Quality Control & Report
# ============================================================

print()
print("=" * 70)
print("MODULE 8 - FINAL QUALITY CONTROL")
print("=" * 70)

from datetime import datetime

# ------------------------------------------------------------
# Verify Processed Dataset
# ------------------------------------------------------------

patient_dirs = sorted([
    p for p in OUTPUT_DIR.iterdir()
    if p.is_dir()
])

total_patients = len(patient_dirs)

t1_count = 0
lesion_count = 0
dmri_count = 0

missing_files = []

for patient in patient_dirs:

    patient_name = patient.name

    # ------------------------
    # T1
    # ------------------------

    t1_exists = (
        (patient / "T1.nii.gz").exists() or
        (patient / "T1.nii").exists()
    )

    # ------------------------
    # Lesion
    # ------------------------

    lesion_exists = (
        (patient / "Lesion.nii.gz").exists() or
        (patient / "Lesion.nii").exists()
    )

    # ------------------------
    # dMRI (optional)
    # ------------------------

    dmri_exists = (
        (patient / "dMRI.nii.gz").exists() or
        (patient / "dMRI.nii").exists()
    )

    if t1_exists:
        t1_count += 1
    else:
        missing_files.append(
            f"{patient_name}: Missing T1"
        )

    if lesion_exists:
        lesion_count += 1
    else:
        missing_files.append(
            f"{patient_name}: Missing Lesion"
        )

    if dmri_exists:
        dmri_count += 1

# ------------------------------------------------------------
# Verify Metadata
# ------------------------------------------------------------

dataset_csv_exists = (
    OUTPUT_DIR / "dataset.csv"
).exists()

dataset_json_exists = (
    OUTPUT_DIR / "dataset.json"
).exists()

processing_log_exists = (
    LOG_DIR / "processing_log.csv"
).exists()

# ------------------------------------------------------------
# Final Report
# ------------------------------------------------------------

report = []

report.append("=" * 60)
report.append("AIMS-TBI FINAL PREPROCESSING REPORT")
report.append("=" * 60)
report.append("")
report.append(f"Generated : {datetime.now()}")
report.append("")
report.append(f"Patients Processed : {total_patients}")
report.append(f"T1 Images          : {t1_count}")
report.append(f"Lesion Masks       : {lesion_count}")
report.append(f"dMRI Images        : {dmri_count}")
report.append("")
report.append(f"dataset.csv        : {dataset_csv_exists}")
report.append(f"dataset.json       : {dataset_json_exists}")
report.append(f"processing_log.csv : {processing_log_exists}")
report.append("")
report.append(f"Missing Files : {len(missing_files)}")

if len(missing_files):

    report.append("")
    report.append("Missing:")

    report.extend(missing_files)

report.append("")
report.append("=" * 60)
report.append("END OF REPORT")
report.append("=" * 60)

report_path = OUTPUT_DIR / "FINAL_REPORT.txt"

with open(report_path, "w") as f:

    f.write("\n".join(report))

# ------------------------------------------------------------
# Console Summary
# ------------------------------------------------------------

print()

print("Patients            :", total_patients)
print("T1 Images           :", t1_count)
print("Lesion Masks        :", lesion_count)
print("dMRI Images         :", dmri_count)

print()

print("dataset.csv         :", dataset_csv_exists)
print("dataset.json        :", dataset_json_exists)
print("processing_log.csv  :", processing_log_exists)

print()

print("Missing Files       :", len(missing_files))

print()

print("Final report saved to")

print(report_path)

print()

print("=" * 70)
print("MODULE 8 COMPLETED")
print("=" * 70)