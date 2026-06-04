# Execution Guide

This document explains how to run the required scripts in the repository.

Currently, this guide covers dataset preprocessing scripts used to extract visual and remote physiological features 
from facial videos. 
---

## 1. Prerequisites

Before running any script, ensure that:

- The conda environment is activated.
- Required Python libraries are installed.
- OpenFace is installed and built successfully.
- The OpenFace `FeatureExtraction` binary is available.
- Dataset videos are placed in the expected directory structure.
- Dataset paths inside the scripts are updated according to your local system.

Activate the conda environment:

```bash
conda activate HCAT-Pain
```

---

## 2. Dataset Preprocessing

Dataset preprocessing consists of extracting:

1. Visual facial features using OpenFace.
2. Remote physiological signals such as pulse and respiration.

The following scripts are used for dataset preprocessing:

1. `extract_openface_features.py`
2. `extract_temporal_signal_files.py`

---

### 2.1 Expected Dataset Structure

The scripts expect the dataset videos to be organized in the following structure:

```text
Dataset/
├── Train/
│   └── video/
│       └── subject_id/
│           └── video.mp4
├── Validation/
│   └── video/
│       └── subject_id/
│           └── video.mp4
```

In general, each video should follow this format:

```text
Dataset/Train/video/subject_id/video.mp4
```

Before running the scripts, verify that the dataset path used inside the scripts points to the correct dataset root directory.

---

### 2.2 Run `extract_openface_features.py`

This script runs OpenFace on facial videos and extracts frame-wise facial descriptors.

The extracted features may include (customizable):

- Action Unit intensities
- Action Unit classifications
- Head pose
- Gaze features
- 2D facial landmarks
- 3D facial landmarks

#### Important paths to check

Before running the script, update the following paths if required:

```python
OPENFACE_BIN = Path("../libraries/OpenFace/build/bin/FeatureExtraction")
base_path = "/path/to/dataset/root"
log_path = "openface_failed_log.txt"
```

Make sure that `OPENFACE_BIN` points to the correct OpenFace `FeatureExtraction` binary.

#### Command

Run the script from the repository root:

```bash
python scripts/extract_openface_features.py
```

After successful execution, OpenFace CSV files should be generated in the configured output directory.

---

### 2.3 Run `extract_temporal_signal_files.py`

This script extracts remote physiological signals from facial videos using Eulerian and Lagrangian techniques.

The extracted signals include:

- Pulse signal
- Respiratory signal

#### Important paths and parameters to check

Before running the script, check and update the following if required:

```python
base_path = "/path/to/dataset/root"
```

#### Command

Run the script from the repository root:

```bash
python scripts/extract_temporal_signal_files.py
```

After successful execution, temporal signal files should be generated for the processed videos.

---

### 2.4 Note

#### Recommended Execution Order

Run the dataset preprocessing scripts in the following order:

```bash
python scripts/extract_openface_features.py
python scripts/extract_temporal_signal_files.py
```

The first script extracts visual facial descriptors using OpenFace.

The second script extracts remote physiological signals from the videos.
