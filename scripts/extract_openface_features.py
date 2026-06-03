import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Limit OpenBLAS threads inherited by OpenFace subprocesses to reduce CPU/memory contention.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from natsort import natsorted
from tqdm import tqdm

# Path to OpenFace FeatureExtraction binary (TODO: You need to change this accordingly)
OPENFACE_BIN = Path("../libraries/OpenFace/build/bin/FeatureExtraction")
log_path = "openface_failed_log.txt"

# Path to the Dataset Videos
base_path = "/media/user/Projects/Datasets/AI4Pain/2024/"

# Collect videos from both Train and Validation splits in natural sorted order.
# Note: Strong assumption on dataset. Please change in case of other dataset and/or dataset structure.
video_paths = natsorted(glob.glob(os.path.join(base_path, "Train", "*", "*", "*.mp4")) +
                        glob.glob(os.path.join(base_path, "Validation", "*", "*", "*.mp4")))

# Extract only required OpenFace features and suppress aligned-frame/visualization outputs.
# if only Action Units are required, use only the "-aus" flag from the second line
# Refer to https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments for additional flags
FLAGS = [
    "-noAlignedOutput", "-noVisualization",
    "-2Dfp", "-3Dfp", "-pose", "-aus", "-gaze"
]

## Use this if you want the output folder to be split-wise.
## Add the subjects to each split accordingly. Ensure split wise subject exclusivity to prevent data-leakage
# subject_list_train = natsorted([])
# subject_list_val = natsorted([])
# subject_list_test = natsorted([])


def process_video(video_path):
    # Use the parent folder name as the subject ID for organizing OpenFace outputs.
    # Note: Strong assumption on dataset. Please change in case of other dataset and/or dataset structure.
    subject = int(video_path.split("/")[-2])

    ## Use this if you want the output folder to be split-wise
    # if subject in subject_list_train:
    #     split = "Train"
    # elif subject in subject_list_val:
    #     split = "Validation"
    # else:
    #     split = "Test"
    # out_dir = os.path.join(base_path, "Extracted_OF", split, str(subject))

    out_dir = os.path.join(base_path, "Extracted_OF_5", str(subject))
    os.makedirs(out_dir, exist_ok=True)

    # Run OpenFace on the current video and save outputs in the subject-specific folder.
    cmd = [str(OPENFACE_BIN), "-f", str(video_path), "-out_dir", str(out_dir)] + FLAGS
    result_ = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result_.stdout.decode()

    # Return the OpenFace console output only when the process fails.
    if result_.returncode != 0:
        return f"Failed: {video_path}\n{output}"
    return None


if __name__ == "__main__":
    # Ensure the OpenFace executable is available before starting batch processing.
    if not OPENFACE_BIN.exists():
        raise FileNotFoundError(f"OpenFace binary not found: {OPENFACE_BIN}")

    # Use a conservative worker count to avoid overloading OpenFace/OpenBLAS and memory;
    # increase the upper limit gradually if the run remains stable.
    max_workers = min(8, (os.cpu_count() or 1) // 2 + 1)  # If stable, increase gradually.

    errors = []

    # Process videos in parallel while preserving the input order returned by executor.map.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_video, video_paths), total=len(video_paths)):
            # process_video returns None on success and an error message on failure.
            if result:
                errors.append(result)

    # Write all failed cases at the end to avoid multiple processes writing to the same log file.
    if errors:
        with open(log_path, "w") as f:
            f.write("\n\n".join(errors))
