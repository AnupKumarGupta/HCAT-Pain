
# Developer Installation Guide

This document provides step-by-step instructions for setting up the development environment for **HCAT-Pain**. This file contains the detailed developer setup instructions, including environment creation, PyTorch installation, OpenFace installation, OpenCV notes, and common troubleshooting steps.

---

## Table of contents

- [1. Tested environment](#1-tested-environment)
- [2. Important note about PyTorch version](#2-important-note-about-pytorch-version)
- [3. Create a conda environment](#3-create-a-conda-environment)
- [4. Install PyTorch](#4-install-pytorch)
- [5. Verify PyTorch and CUDA](#5-verify-pytorch-and-cuda)
- [6. Install OpenFace](#6-install-openface)
  - [6.1 Official OpenFace installation guide](#61-official-openface-installation-guide)
  - [6.2 OpenCV 4.1.0 installation note](#62-opencv-410-installation-note)
  - [6.3 Fix OpenCV permission issues](#63-fix-opencv-permission-issues)
  - [6.4 Fix dlib download issue](#64-fix-dlib-download-issue)
  - [6.5 Verify OpenFace installation](#65-verify-openface-installation)

---

## 1. Tested environment

The following setup was used during development:

| Component | Version / Setting |
|-----------|-------------------|
| Conda environment name | `HCAT-Pain` |
| Python | `3.8` |
| PyTorch | `1.12.1` |
| Torchvision | `0.13.1` |
| Torchaudio | `0.12.1` |
| CUDA toolkit | `11.3` |
| OpenFace | Built from source |

---

## 2. Important note about PyTorch version

The original experiments were run using **PyTorch 1.12.1 with CUDA 11.3**.

Newer PyTorch versions may also work, but they have not been verified for this repository.

> [!NOTE]
> The older PyTorch version was used because the original development was done on an older system. Unless you are intentionally updating and testing the full pipeline, it is recommended to first reproduce the environment using the versions listed above.

---

## 3. Create a conda environment

Creating a separate conda environment is recommended.

```bash
conda create -n HCAT-Pain python=3.8
conda activate HCAT-Pain
```

To confirm that the environment is active:

```bash
conda info --envs
```

The active environment should be marked with `*`.

---

## 4. Install PyTorch

Install PyTorch, Torchvision, Torchaudio, and CUDA toolkit using the following command:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

This installs the PyTorch version used during development.

---

## 5. Verify PyTorch and CUDA

After installation, verify that PyTorch is installed correctly and that CUDA is available:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

The expected output is:

```text
1.12.1
11.3
True
```

If the third line prints:

```text
False
```

then PyTorch is not able to access the GPU. In that case, check:

- whether the NVIDIA driver is installed correctly,
- whether the correct conda environment is active,
- whether the installed PyTorch build supports CUDA,
- whether the machine has access to a CUDA-compatible GPU.

---

## 6. Install OpenFace

OpenFace is required for extracting facial Action Unit features and related facial descriptors.

Installing OpenFace from source can take some time. Follow the official Unix installation guide carefully.

### 6.1 Official OpenFace installation guide

Official OpenFace installation guide:

- Dependency installation:  
  https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation#dependency-installation

- Actual OpenFace installation:  
  https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation#actual-openface-installation

> [!IMPORTANT]
> During OpenFace installation, make sure that the **OpenCV 4.1.0 compilation step** is completed properly.

OpenCV 4.1.0 installation section:

https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation#actual-openface-installation:~:text=Download%20and%20compile%20OpenCV%204.1.0

---

### 6.2 OpenCV 4.1.0 installation note

During OpenFace installation, OpenCV 4.1.0 is downloaded and compiled.

This step is important. Do not skip it.

The OpenCV build process may fail if the downloaded OpenCV directory is owned by `root` or if the current user does not have permission to create build files.

---

### 6.3 Fix OpenCV permission issues

If the OpenCV folder does not allow you to create directories or build files, first move inside the OpenCV directory.

For example:

```bash
pwd
```

The output may look like this:

```text
../../OpenFace/opencv-4.1.0
```

Now check the directory ownership:

```bash
ls -ld .
```

The output may look like this:

```text
drwxr-xr-x 11 root root 4096 Apr 7 2019 .
```

If the directory is owned by `root`, change the ownership to the current user:

```bash
sudo chown -R $USER:$USER .
```

Check the ownership again:

```bash
ls -ld .
```

After ownership is fixed, the directory should be writable by the current user.

> [!NOTE]
> The exact date and size shown by `ls -ld .` may differ on your system. What matters is that the directory should not remain owned by `root root` if you need to build files inside it as a normal user.

---

### 6.4 Fix dlib download issue

During OpenFace installation, the official instructions may use a command similar to:

```bash
wget http://dlib.net/files/dlib-19.13.tar.bz2;
```

If this command fails, run the command without the trailing semicolon:

```bash
wget http://dlib.net/files/dlib-19.13.tar.bz2
```

Then continue with the remaining OpenFace installation steps.

---

### 6.5 Verify OpenFace installation

Before assuming that OpenFace has installed correctly, verify that the `FeatureExtraction` binary exists.

Depending on where OpenFace was cloned and built, it is usually located at a path similar to:

```text
OpenFace/build/bin/FeatureExtraction
```

You can check it using:

```bash
ls OpenFace/build/bin/FeatureExtraction
```

If the file exists, the OpenFace binary is available.

You can also check it from inside the OpenFace directory:

```bash
ls build/bin/FeatureExtraction
```