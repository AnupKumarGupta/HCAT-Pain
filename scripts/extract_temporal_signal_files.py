import math
import os
from glob import glob

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from einops import rearrange, reduce
from matplotlib import pyplot as plt
from natsort import natsorted
from enum import Enum

from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.signal import butter, filtfilt, detrend
from utils import openface_indices, epsilon, SUBJECT_LIST_VALIDATION, SPLIT_USED

# Initialize MediaPipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Define landmark indices for various facial regions
mp_left_eye_indices = [55, 65, 52, 53, 46, 124, 31, 228, 229, 230, 231, 232, 233, 244, 189]
mp_right_eye_indices = [285, 295, 282, 283, 276, 353, 261, 448, 449, 450, 451, 452, 453, 464, 413]
mp_lip_indices = [186, 92, 165, 167, 164, 57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322, 391, 393, 164]
mp_forehead_indices = [54, 103, 67, 109, 10, 338, 297, 332, 284]
mp_nose_indices = ([1, 2, 4, 5, 6, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440] +
                   [3, 19, 51, 122, 131, 134, 196, 198, 209, 236, 248, 281, 351, 360, 363, 419, 420, 429, 456])
# Adds extra landmark indices on the edge of the nose. Increased the points for robust LM tracking
mp_right_cheeks = [36, 50, 100, 117, 118, 119, 120, 123, 142, 187, 205]
mp_left_cheeks = [266, 280, 329, 346, 347, 348, 349, 352, 371, 411, 425]

DEFAULT_FPS = 30.0
SKIP_SECONDS = 0.5
EXTEND_FOREHEAD = False
DISPLAY_INTERMEDIATE_OUTPUTS = False
USE_ADAPTIVE_BLOCK_SIZE = True
PERCENTAGE_OF_FACE_TO_BE_USED_AS_BLOCK = 10
USE_SKIN_SEGMENTATION = False
MAX_LANDMARK_START_FRAME = 10
INCLUDE_CHEEK_LMS_FOR_LAGRANGIAN = True
MIN_LAGRANGIAN_POINTS = 20
# We searched empirically and qualitatively and came up the config values. Tweaking further may be done by users
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class VideoType(Enum):
    BASELINE = 0
    PAIN_HIGH = 1
    PAIN_LOW = 2
    REST = 3


time_duration_for_video_type = {
    VideoType.BASELINE: 60,
    VideoType.PAIN_HIGH: 10,
    VideoType.PAIN_LOW: 10,
}


def display_image(title, image, save_image=False):
    """Display an image using matplotlib.

    Args:
        title (str): Title for the displayed image.
        image (ndarray): The image to display.
        save_image (bool): If True, saves the image as a PNG file.
    """
    plt.figure(figsize=(6, 6))
    if len(image.shape) == 3:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:  # Grayscale image
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

    if save_image:
        cv2.imwrite(title + ".png", image)


def extract_frames_from_video(video_path):
    """Extract frames from an input video file.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: List of frames extracted from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def compute_convex_hull(landmarks, frame, display_steps=False):
    """Compute the convex hull from facial landmarks.

    Args:
        landmarks (list): List of (x, y) tuples representing landmarks.
        frame (ndarray): The original frame to display the hull on.
        display_steps (bool): If True, displays the convex hull.

    Returns:
        ndarray: Points of the convex hull.
    """
    hull = cv2.convexHull(np.array(landmarks, dtype=np.int32))
    if display_steps:
        hull_frame = frame.copy()
        cv2.polylines(hull_frame, [hull], isClosed=True, color=(255, 0, 0), thickness=2)
        display_image("Convex Hull", hull_frame)
    return hull


def apply_mask(frame, hull, display_steps=False):
    """Apply a mask to isolate the facial region.

    Args:
        frame (ndarray): The original frame.
        hull (ndarray): Points of the convex hull.
        display_steps (bool): If True, displays the masked frame.

    Returns:
        ndarray: The masked frame isolating the facial region.
    """
    mask = np.zeros_like(frame)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))  # Fill the hull area
    masked_frame = cv2.bitwise_and(frame, mask)  # Apply mask
    if display_steps:
        display_image("Masked Region", masked_frame)
    return masked_frame


def crop_facial_region(masked_frame, hull, display_steps=False):
    """Crop the facial region from the masked frame.

    Args:
        masked_frame (ndarray): The masked frame.
        hull (ndarray): Points of the convex hull.
        display_steps (bool): If True, displays the cropped region.

    Returns:
        ndarray: The cropped facial region.
    """
    x, y, w, h = cv2.boundingRect(hull)
    cropped_face = masked_frame[y:y + h, x:x + w]
    if display_steps:
        display_image("Cropped Facial Region", cropped_face)
    return cropped_face


def remove_eye_and_lip_regions(cropped_face, landmarks, display_steps=False):
    """Remove the eye and lip regions from the cropped facial region by filling them with zeros.

    Args:
        cropped_face (ndarray): The cropped face image.
        landmarks (list): The list of all detected facial landmarks.
        display_steps (bool): Whether to display the intermediate steps or not.

    Returns:
        ndarray: The cropped face with eye and lip regions removed.
    """
    # Extract landmarks for the left eye and right eye
    left_eye_landmarks = np.array([landmarks[i] for i in mp_left_eye_indices], dtype=np.int32)
    right_eye_landmarks = np.array([landmarks[i] for i in mp_right_eye_indices], dtype=np.int32)
    lip_landmarks = np.array([landmarks[i] for i in mp_lip_indices], dtype=np.int32)

    # Create a mask for the eye regions
    mask = np.zeros_like(cropped_face)

    # Create convex hulls for the left and right eye regions
    cv2.fillConvexPoly(mask, cv2.convexHull(left_eye_landmarks), (255, 255, 255))
    cv2.fillConvexPoly(mask, cv2.convexHull(right_eye_landmarks), (255, 255, 255))
    cv2.fillConvexPoly(mask, cv2.convexHull(lip_landmarks), (255, 255, 255))

    # Invert the mask (so that eye regions are zero and everything else is 1)
    mask_inv = cv2.bitwise_not(mask)

    if display_steps:
        display_image("mask", mask)

    if display_steps:
        display_image("mask_inv", mask_inv)

    # Apply the mask to remove the eye regions by setting the eye region to black (0)
    cropped_face_with_no_eyes_and_lips = cv2.bitwise_and(cropped_face, mask_inv)

    if display_steps:
        display_image("Cropped Facial Region with Eyes Removed", cropped_face_with_no_eyes_and_lips)

    return cropped_face_with_no_eyes_and_lips


def butter_bandpass_filter(data, lowcut=0.7, highcut=4.2, fs=30.0, order=4):
    """Apply a Butterworth bandpass filter to the input data.

    Args:
        data (ndarray): Input signal (2D array with shape rois x frames).
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): The order of the filter.

    Returns:
        ndarray: Filtered data with the same shape as the input.
    """

    # Ensure data is 2D: If the input is 1D, convert it to a 2D array with one row.
    if data.ndim == 1:
        data = data[np.newaxis, :]
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


def get_lowcut_highcut_frequencies_based_on_physiological_parameter(physiological_parameter):
    """
    physiological_parameter (string): Type of physiological parameter to be extracted ("hr" or "rr")
    """
    assert physiological_parameter in ["hr", "rr"], "physiological_parameter must be either 'hr', or 'rr'"
    lowcut = 0
    highcut = 0

    if physiological_parameter == "hr":
        lowcut = 0.7  # 0.7
        highcut = 4.2  # 4.2
    elif physiological_parameter == "rr":
        lowcut = 0.083
        highcut = 0.5
    return lowcut, highcut


def mean_pool_blocks(cropped_face, block_size=16, display_steps=False):
    """Apply mean pooling to non-overlapping blocks of the image.

    Args:
        cropped_face (ndarray): Input RGB image (height, width, channels).
        block_size (int): Size of the square block for pooling.
        display_steps (bool): Whether to display the output of intermediate steps.

    Returns:
        ndarray: Downsampled image with mean-pooled blocks.
    """
    # Ensure the image dimensions are divisible by block_size
    h, w, c = cropped_face.shape
    h_trimmed = h - (h % block_size)
    w_trimmed = w - (w % block_size)
    cropped_face_trimmed = cropped_face[:h_trimmed, :w_trimmed, :]

    pooled_image = reduce(
        cropped_face_trimmed * 1.0,
        "(h h1) (w w1) c -> h w c",
        reduction="mean",
        h1=block_size,
        w1=block_size
    )

    if display_steps:
        display_image("Cropped Face", cropped_face)
        display_image("Cropped Face Trimmed", cropped_face_trimmed)
        display_image("Pooled Image", np.round(pooled_image).astype("uint8"))

    return pooled_image


def detect_landmarks(frame, display_steps=False):
    """Detect facial landmarks using MediaPipe Face Mesh.

    Args:
        frame (ndarray): The input frame from the video.
        display_steps (bool): If True, displays the frame with landmarks.

    Returns:
        list: List of detected landmarks as (x, y) tuples.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        converted_landmarks = []
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            if EXTEND_FOREHEAD:
                if idx in mp_forehead_indices:
                    y = max(0, y - 55)  # Adjust forehead landmark position
            converted_landmarks.append((x, y))

        # Get face bounding box from landmarks
        xs = [point[0] for point in converted_landmarks]
        ys = [point[1] for point in converted_landmarks]
        face_x_min = min(xs)
        face_x_max = max(xs)
        face_y_min = min(ys)
        face_y_max = max(ys)
        face_width = face_x_max - face_x_min
        face_height = face_y_max - face_y_min

        if display_steps:
            frame_copy = frame.copy()
            cv2.rectangle(
                frame_copy,
                (face_x_min, face_y_min),
                (face_x_max, face_y_max),
                (255, 0, 0),
                2
            )
            for landmark in converted_landmarks:
                cv2.circle(frame_copy, landmark, 1, (0, 255, 0), -1)  # Draw landmarks
            display_image("Landmark Detection", frame_copy)

        return converted_landmarks, face_height, face_width
    return None, 0, 0


def get_landmarks_from_succeeding_frames(video_frames, display_steps):
    """Attempt to detect landmarks in succeeding frames if the first frame fails.

    Args:
        video_frames (list): List of video frames.
        display_steps (bool): Whether to display intermediate steps.

    Returns:
        list: Detected landmarks from the succeeding frames or None if not found.
    """
    for frame in video_frames:
        landmarks, height, width = detect_landmarks(frame, display_steps)
        if landmarks is None:
            continue
        else:
            return landmarks, height, width
    return None, 0, 0


def get_landmarks_from_succeeding_frames_with_index(video_frames, display_steps):
    """Attempt to detect landmarks in succeeding frames if the first frame fails.

    Args:
        video_frames (list): List of video frames.
        display_steps (bool): Whether to display intermediate steps.

    Returns:
        list: Detected landmarks from the succeeding frames or None if not found.
    """
    for frame_index, frame in enumerate(video_frames):
        landmarks, height, width = detect_landmarks(frame, display_steps)
        if landmarks is None:
            continue
        else:
            return landmarks, frame_index
    return None, -1


def display_nose_landmarks(frame, landmarks, nose_indices, show_indices=False):
    """Display selected nose landmarks on the original frame.

    Args:
        frame (ndarray): Original BGR frame.
        landmarks (list): MediaPipe landmarks as (x, y) coordinates.
        nose_indices (list): Landmark indices corresponding to the nose region.
        show_indices (bool): If True, writes landmark index beside each point.
    """
    frame_copy = frame.copy()

    for idx in nose_indices:
        x, y = landmarks[idx]

        cv2.circle(frame_copy, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        if show_indices:
            cv2.putText(frame_copy, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
                        cv2.LINE_AA)
    display_image("Nose Landmarks", frame_copy)


def get_lagrangian_points_for_lk(landmarks, lagrangian_indices, frame_shape):
    """Convert selected MediaPipe landmarks into Lucas-Kanade input points.

    Args:
        landmarks (list): List of facial landmarks as (x, y) pixel coordinates.
        lagrangian_indices (list): Landmark indices to be tracked.
        frame_shape (tuple): Shape of the frame, usually frame.shape.

    Returns:
        ndarray or None: Points with shape N x 1 x 2 and dtype np.float32.
    """
    frame_height, frame_width = frame_shape[:2]
    points = []
    for idx in lagrangian_indices:
        if idx >= len(landmarks):
            continue

        x, y = landmarks[idx]

        if 0 <= x < frame_width and 0 <= y < frame_height:
            points.append([x, y])

    if len(points) < MIN_LAGRANGIAN_POINTS:
        return None
    points = np.array(points, dtype=np.float32)
    points = points.reshape(-1, 1, 2)
    return points


def skin_ratio_in_roi_bgr(roi_bgr):
    """Compute the fraction of skin-like pixels in a BGR ROI.

    Args:
        roi_bgr (ndarray): ROI image in BGR format, shape H x W x 3.

    Returns:
        float: Fraction of skin-like pixels in [0, 1].
    """
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    skin_mask = (
            (cb >= 75) & (cb <= 135) &
            (cr >= 130) & (cr <= 180)
    )

    return np.sum(skin_mask) / skin_mask.size


def retain_rois_with_only_skin_region(cropped_face, block_size, min_skin_ratio=0.90, display_steps=False):
    """Retain only those non-overlapping patches that mostly contain skin pixels.

    Args:
        cropped_face (ndarray): Cropped BGR face image after masking eyes and lips.
        block_size (int): Size of each square patch.
        min_skin_ratio (float): Minimum skin-pixel ratio required to retain a patch.
        display_steps (bool): Whether to display intermediate results.

    Returns:
        ndarray: Cropped face with non-skin patches set to zero.
    """
    h, w, _ = cropped_face.shape

    h_trimmed = h - (h % block_size)
    w_trimmed = w - (w % block_size)

    cropped_face_trimmed = cropped_face[:h_trimmed, :w_trimmed].copy()
    retained_face = np.zeros_like(cropped_face_trimmed)

    for y in range(0, h_trimmed, block_size):
        for x in range(0, w_trimmed, block_size):
            roi = cropped_face_trimmed[y:y + block_size, x:x + block_size]

            ratio = skin_ratio_in_roi_bgr(roi)

            if ratio >= min_skin_ratio:
                retained_face[y:y + block_size, x:x + block_size] = roi

    if display_steps:
        display_image("Cropped Face Before Skin ROI Retention", cropped_face_trimmed)
        display_image("Cropped Face After Skin ROI Retention", retained_face)

    return retained_face


def get_temporal_signals_from_video(video_frames, block_size=16, display_steps=False):
    """Process the input video frames for rPPG signal extraction.

    Args:
        video_frames (list or ndarray): List of frames from the video.
        block_size (int): Size of the block for mean pooling.
        display_steps (bool): Whether to display intermediate processing steps.

    Returns:
        list: List of average pixel values across blocks for each frame.
    """
    signals = []
    first_frame_flag = True

    for frame in video_frames:
        if first_frame_flag:
            # Detect landmarks in the first frame
            landmarks, height, width = detect_landmarks(frame, display_steps)
            video_frames_copy = np.copy(video_frames)

            if landmarks is None:
                print("No landmarks detected for first frame. Attempting to detect landmark for succeeding frames.")
                landmarks, height, width = get_landmarks_from_succeeding_frames(video_frames_copy, display_steps)

            if landmarks is None:
                print("No landmarks detected for the clip.")
                return None
            first_frame_flag = False
            if USE_ADAPTIVE_BLOCK_SIZE:
                block_size = math.ceil((PERCENTAGE_OF_FACE_TO_BE_USED_AS_BLOCK / 100) * max(height, width))

        # Process each frame
        hull = compute_convex_hull(landmarks, frame, display_steps)
        masked_frame = apply_mask(frame, hull, display_steps)
        masked_face_no_eyes_lips = remove_eye_and_lip_regions(masked_frame, landmarks, display_steps)
        cropped_face = crop_facial_region(masked_face_no_eyes_lips, hull, display_steps)
        if USE_SKIN_SEGMENTATION:
            rois_retained_with_only_skin_regions = retain_rois_with_only_skin_region(cropped_face, block_size, 0.85)
            mean_pooled_image = mean_pool_blocks(rois_retained_with_only_skin_regions, block_size, display_steps)
        else:
            mean_pooled_image = mean_pool_blocks(cropped_face, block_size, display_steps)
        average_pixel_values = rearrange(mean_pooled_image, 'h w c -> (h w) c')
        signals.append(average_pixel_values)

    return signals


def get_lagrangian_temporal_signals_from_video(video_frames, display_steps=False):
    """Process the input video frames for Lagrangian respiratory signal extraction.

    Args:
        video_frames (list or ndarray): List of frames from the video.
        display_steps (bool): Whether to display intermediate processing steps.

    Returns:
        ndarray or None: Lagrangian motion trajectories from selected facial points.
    """
    frame_idx = 0

    # Choose points for Lagrangian tracking
    if INCLUDE_CHEEK_LMS_FOR_LAGRANGIAN:
        lagrangian_indices = mp_nose_indices + mp_right_cheeks + mp_left_cheeks
    else:
        lagrangian_indices = mp_nose_indices

    # Try detecting landmarks in the first frame
    landmarks, _, _ = detect_landmarks(video_frames[0], display_steps)

    # If first-frame detection fails, try succeeding frames
    if landmarks is None:
        print("No landmarks detected for first frame. Attempting to detect landmark for succeeding frames.")

        landmarks, frame_idx = get_landmarks_from_succeeding_frames_with_index(
            video_frames,
            display_steps
        )

    # If no landmarks are detected anywhere, skip this clip
    if landmarks is None:
        print("No landmarks detected for the clip.")
        return None

    # If landmarks are detected too late, skip this clip
    if frame_idx > MAX_LANDMARK_START_FRAME:
        print("No landmarks detected for the clip within the first 10 frames. Skipping the video.")
        return None

    if display_steps:
        display_nose_landmarks(video_frames[frame_idx], landmarks, lagrangian_indices)

    # Convert selected landmarks to Lucas-Kanade input format
    initial_points = get_lagrangian_points_for_lk(
        landmarks=landmarks,
        lagrangian_indices=lagrangian_indices,
        frame_shape=video_frames[frame_idx].shape
    )

    if initial_points is None:
        print("Too few valid Lagrangian landmark points. Skipping the clip.")
        return None

    # Use the frame where landmarks were successfully detected as the tracking start frame.
    # If landmarks were detected in the first frame, frame_idx = 0.
    # If fallback was used, frame_idx may be a later frame.
    start_frame = video_frames[frame_idx]
    start_gray = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)

    prev_gray = start_gray
    prev_points = initial_points

    # Extract the initial y-coordinate of each tracked point.
    initial_y = initial_points[:, 0, 1].copy()

    y_trajectories = []

    # Initialize each trajectory with its starting y-position.
    # If tracking starts from frame 0, each trajectory starts with one value.
    # If tracking starts from a later frame, fill earlier frames with the initial y-value.
    for y in initial_y:
        y_trajectories.append([y] * (frame_idx + 1))

    for current_frame in video_frames[frame_idx + 1:]:
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray,
                                                             prev_points, None, **LK_PARAMS)

        if current_points is None or status is None:
            print("Lucas-Kanade tracking failed.")
            return None

        status = status.reshape(-1).astype(bool)

        # If too few points survive tracking, the motion signal is unreliable.
        if np.sum(status) < MIN_LAGRANGIAN_POINTS:
            print("Too few points tracked successfully. Skipping the clip.")
            return None

        current_points = current_points[status]

        # Remove trajectories corresponding to points that failed tracking.
        y_trajectories = [
            trajectory for trajectory, is_valid in zip(y_trajectories, status)
            if is_valid
        ]

        for trajectory, point in zip(y_trajectories, current_points):
            y = point[0, 1]
            trajectory.append(y)

        prev_gray = current_gray
        prev_points = current_points.reshape(-1, 1, 2)

    y_trajectories = np.array(y_trajectories, dtype=np.float32)

    # Final safety check. If too few trajectories remain, skip the clip.
    if y_trajectories.shape[0] < MIN_LAGRANGIAN_POINTS:
        print("Too few final Lagrangian trajectories. Skipping the clip.")
        return None

    # Convert absolute y-coordinates to vertical displacements.
    # For each trajectory, subtract its own first y-coordinate.
    vertical_displacement_trajectories = (y_trajectories - y_trajectories[:, [0]])
    return vertical_displacement_trajectories


def compute_temporal_signals_non_overlapping_clips(video, destination_path, FPS=35., skip_seconds=10, clip_seconds=10,
                                                   block_size=16, video_info=None, physiological_parameter="hr",
                                                   display_steps=False):
    """Process non-overlapping clips from the given video, skipping specified seconds at the start and end.

    Args:
        video (list): List of frames extracted from the video.
        destination_path (string): The path where extracted signals is to stored
        FPS (float/int): Frames per second of the video (default is 35).
        skip_seconds (float): Number of seconds to skip at the start and end of the video (default is 10).
        clip_seconds (int): Duration in seconds of each clip to be processed (default is 10).
        block_size (int): Size of the block for mean pooling.
        video_info (dict): Metadata about the video for saving purposes.
        physiological_parameter (string): Type of physiological parameter to be extracted ("hr" or "rr")
        display_steps (bool): Whether to display intermediate processing steps.
    """
    assert physiological_parameter in ["hr", "rr"], "physiological_parameter must be either 'hr', or 'rr'"

    skip_frame_count = math.floor(FPS * skip_seconds)
    clip_frame_count = math.floor(FPS * clip_seconds)

    start_frame = skip_frame_count
    end_frame = len(video) - skip_frame_count

    for clip_number, i in enumerate(range(start_frame, end_frame, clip_frame_count)):
        clip = video[i:i + clip_frame_count]

        if len(clip) >= clip_frame_count:
            temporal_signal = get_temporal_signals_from_video(clip, block_size=block_size, display_steps=display_steps)
            if temporal_signal is None:
                print(f"Issue with temporal signal {video_info}, clip # {clip_number}")
            else:
                save_temporal_signal(temporal_signal, clip_number, video_info, physiological_parameter,
                                     destination_path, FPS)


def compute_lagrangian_signals_non_overlapping_clips(video, destination_path, FPS=35., skip_seconds=10, clip_seconds=10,
                                                     video_info=None, physiological_parameter="hr",
                                                     display_steps=False):
    """Process non-overlapping clips from the given video to obtain motion signals,
    skipping specified seconds at the start and end.

    Args:
        video (list): List of frames extracted from the video.
        destination_path (string): The path where extracted signals is to stored
        FPS (float/int): Frames per second of the video (default is 35).
        skip_seconds (float): Number of seconds to skip at the start and end of the video (default is 10).
        clip_seconds (int): Duration in seconds of each clip to be processed (default is 10).
        video_info (dict): Metadata about the video for saving purposes.
        physiological_parameter (string): Type of physiological parameter to be extracted ("hr" or "rr")
        display_steps (bool): Whether to display intermediate processing steps.
    """
    assert physiological_parameter in ["hr", "rr"], "physiological_parameter must be either 'hr', or 'rr'"

    skip_frame_count = math.floor(FPS * skip_seconds)
    clip_frame_count = math.floor(FPS * clip_seconds)

    start_frame = skip_frame_count
    end_frame = len(video) - skip_frame_count

    for clip_number, i in enumerate(range(start_frame, end_frame, clip_frame_count)):
        clip = video[i:i + clip_frame_count]

        if len(clip) >= clip_frame_count:
            temporal_signal = get_lagrangian_temporal_signals_from_video(clip, display_steps=display_steps)
            if temporal_signal is None:
                print(f"Issue with temporal signal {video_info}, clip # {clip_number}")
            else:
                save_lagrangian_temporal_signal(temporal_signal, clip_number, video_info, physiological_parameter,
                                                destination_path, FPS)


def save_lagrangian_temporal_signal(temporal_signal, clip_number, video_info, physiological_parameter,
                                    destination_path, FPS):
    """Save Lagrangian vertical motion trajectories to disk.

        Args:
            temporal_signal (ndarray): Lagrangian trajectories with shape points x frames.
            clip_number (int): Clip number being processed.
            video_info (dict): Metadata for naming directories.
            physiological_parameter (str): "hr" or "rr", used to choose filtering band.
            destination_path (str): Root output directory.
            FPS (float): Original video FPS.
        """
    assert physiological_parameter in ["hr", "rr"], "physiological_parameter must be either 'hr' or 'rr'"

    temporal_signal = np.asarray(temporal_signal)

    if temporal_signal.ndim != 2:
        print(f"Invalid Lagrangian signal shape: {temporal_signal.shape}")
        return

    if temporal_signal.shape[0] < MIN_LAGRANGIAN_POINTS:
        print(f"Too few Lagrangian trajectories: {temporal_signal.shape[0]}")
        return

    lowcut, highcut = get_lowcut_highcut_frequencies_based_on_physiological_parameter(physiological_parameter)

    temporal_signal_filtered = detrend(butter_bandpass_filter(temporal_signal, lowcut, highcut, fs=FPS), axis=1)

    subject = video_info["subject"]
    label = video_info["label"]
    clip_seconds = video_info["clip_seconds"]
    split = video_info["split"]
    session_num = video_info["session_num"]

    if SPLIT_USED and split is not None:
        folder_structure = os.path.join(destination_path, "rppg_signals_lagrangian", physiological_parameter,
                                        f"{clip_seconds:03}s", split, label, subject)
    else:
        folder_structure = os.path.join(destination_path, "rppg_signals_lagrangian", physiological_parameter,
                                        f"{clip_seconds:03}s", label, subject)
    os.makedirs(folder_structure, exist_ok=True)

    if session_num < 0:
        file_name = f"{clip_number:02}.npy"
    else:
        file_name = f"{session_num:02}_{clip_number:02}.npy"

    # Interpolate to DEFAULT_FPS for consistency with Eulerian signals.
    t_old = np.linspace(0, clip_seconds, temporal_signal_filtered.shape[1])
    t_new = np.linspace(0, clip_seconds, int(clip_seconds * DEFAULT_FPS))

    interpolated_signal = interp1d(t_old, temporal_signal_filtered, kind="cubic", axis=1)(t_new)
    np.save(os.path.join(folder_structure, file_name), interpolated_signal)


def save_temporal_signal(temporal_signal, clip_number, video_info, physiological_parameter, destination_path,
                         FPS_video):
    """Save the extracted temporal signals to disk.

    Args:
        temporal_signal (ndarray): Temporal signal data for the clip.
        clip_number (int): The clip number being processed.
        video_info (dict): Metadata about the video for naming directories.
        physiological_parameter (string): Type of physiological parameter to be extracted ("hr" or "rr")
        destination_path (string): The path where extracted signals is to stored
    """
    assert physiological_parameter in ["hr", "rr"], "physiological_parameter must be either 'hr', or 'rr'"

    bgr_signals = rearrange(temporal_signal, "frames rois channels -> channels rois frames")
    roi_masks = reduce(bgr_signals, "channels rois frames -> rois", reduction="mean") >= 20.0
    filtered_bgr_signals = bgr_signals[:, roi_masks, :]

    red_signals = filtered_bgr_signals[2]
    green_signals = filtered_bgr_signals[1]
    blue_signals = filtered_bgr_signals[0]

    lowcut, highcut = get_lowcut_highcut_frequencies_based_on_physiological_parameter(physiological_parameter)

    red_signals_filtered = detrend(butter_bandpass_filter(red_signals, lowcut, highcut, fs=FPS_video), axis=1)
    green_signals_filtered = detrend(butter_bandpass_filter(green_signals, lowcut, highcut, fs=FPS_video), axis=1)
    blue_signals_filtered = detrend(butter_bandpass_filter(blue_signals, lowcut, highcut, fs=FPS_video), axis=1)

    average_signals = (red_signals_filtered + green_signals_filtered + blue_signals_filtered) / 3
    average_signals_filtered = detrend(butter_bandpass_filter(average_signals, lowcut, highcut, fs=FPS_video), axis=1)

    weighted_average_signals = (0.299 * red_signals_filtered + 0.587 * green_signals_filtered +
                                0.114 * blue_signals_filtered)
    weighted_average_signals_filtered = detrend(butter_bandpass_filter(weighted_average_signals, lowcut, highcut,
                                                                       fs=FPS_video), axis=1)

    eta = 3 * red_signals_filtered - 2 * green_signals_filtered
    mu = 1.5 * red_signals_filtered + green_signals_filtered - 1.5 * blue_signals_filtered
    alpha = np.std(eta, axis=1) / (np.std(mu, axis=1) + 1e-8)
    alpha = alpha[:, np.newaxis]
    chrominance_signals = eta - alpha * mu
    chrominance_signals_filtered = detrend(butter_bandpass_filter(chrominance_signals, lowcut, highcut,
                                                                  fs=FPS_video), axis=1)

    subject = video_info["subject"]
    label = video_info["label"]
    clip_seconds = video_info["clip_seconds"]
    block_size = "adaptive_" if USE_ADAPTIVE_BLOCK_SIZE else f"{video_info['block_size']:03}"
    split = video_info["split"]
    session_num = video_info["session_num"]

    # Define the directory structure for saving the signals
    if SPLIT_USED and split is not None:
        folder_structure = os.path.join(destination_path, "rppg_signals", physiological_parameter,
                                        f"{clip_seconds:03}s", f"{block_size}bs", split, label, subject)
    else:
        folder_structure = os.path.join(destination_path, "rppg_signals", physiological_parameter,
                                        f"{clip_seconds:03}s", f"{block_size}bs", label, subject)

    for sub_folder in ["red", "green", "blue", "avg", "w_avg", "chrom"]:
        os.makedirs(os.path.join(folder_structure, sub_folder), exist_ok=True)

    if session_num < 0:
        file_name = f"{clip_number:02}.npy"
    else:
        file_name = f"{session_num:02}_{clip_number:02}.npy"

    name_signal_pair = {
        "red": red_signals_filtered,
        "green": green_signals_filtered,
        "blue": blue_signals_filtered,
        "avg": average_signals_filtered,
        "w_avg": weighted_average_signals_filtered,
        "chrom": chrominance_signals_filtered,
    }

    for name, signal in name_signal_pair.items():
        t_old = np.linspace(0, clip_seconds, green_signals_filtered.shape[1])
        t_new = np.linspace(0, clip_seconds, int(clip_seconds * DEFAULT_FPS))
        interpolated_signal = interp1d(t_old, signal, kind='cubic')(t_new)
        np.save(os.path.join(folder_structure, name, file_name), interpolated_signal)


def get_details_from_video_path(video_path):
    video_name = os.path.basename(video_path).split('.')[0].lower()
    subject_ = video_name.split('_')[0]

    if "baseline" in video_name:
        session_num_ = -1  # No separate sessions were done for baseline
        video_type_ = VideoType.BASELINE
    elif "pain" in video_name:
        _, _, video_type_, session_num_ = video_name.split('_')
        if video_type_ == "high":
            video_type_ = VideoType.PAIN_HIGH
        else:
            video_type_ = VideoType.PAIN_LOW
    else:  # rest
        session_num_ = -1  # Separate sessions were done for rest. However, we do not consider them.
        video_type_ = VideoType.REST

    if SPLIT_USED:
        if "Train" in video_path:
            if int(subject_) in SUBJECT_LIST_VALIDATION:
                split_ = "Validation"
            else:
                split_ = "Train"
        else:
            split_ = "Test"
    else:
        split_ = None
    return subject_, video_type_, split_, int(session_num_)


# def get_details_from_visual_feature_path(csv_file_path):
#     csv_name = os.path.basename(csv_file_path).split('.')[0].lower()
#     subject_ = csv_name.split('_')[0]
#
#     if "baseline" in csv_name:
#         session_num_ = -1  # No separate sessions were done for baseline
#         video_type_ = VideoType.BASELINE
#     elif "pain" in csv_name:
#         _, _, video_type_, session_num_ = csv_name.split('_')
#         if video_type_ == "high":
#             video_type_ = VideoType.PAIN_HIGH
#         else:
#             video_type_ = VideoType.PAIN_LOW
#     else:  # rest
#         session_num_ = -1  # Separate sessions were done for rest. However, we do not consider them.
#         video_type_ = VideoType.REST
#
#     if "Train" in csv_file_path:
#         split_ = "Train"
#     elif "Validation" in csv_file_path:
#         split_ = "Validation"
#     elif "Test" in csv_file_path:
#         split_ = "Test"
#     else:
#         split_ = None
#
#     return subject_, video_type_, split_, int(session_num_)


# def get_details_from_rppg_path(rppg_file_path):
#     if "Train" in rppg_file_path:
#         split_ = "Train"
#     elif "Test" in rppg_file_path:
#         split_ = "Test"
#     else:
#         split_ = "Validation"
#
#     file_components = rppg_file_path.lower().split('/')
#     file_name = os.path.basename(rppg_file_path).split('.')[0].lower()
#     subject_ = file_components[-3]
#
#     if "baseline" == file_components[-4]:
#         session_num_ = -1  # No separate sessions were done for baseline
#         clip_num_ = file_name
#         video_type_ = VideoType.BASELINE
#     elif "pain" in file_components[-4]:
#         session_num_, clip_num_ = file_name.split('_')
#         if "high" in file_components[-4]:
#             video_type_ = VideoType.PAIN_HIGH
#         else:
#             video_type_ = VideoType.PAIN_LOW
#     else:  # rest
#         session_num_ = -1  # Separate sessions were done for rest. However, we do not consider them.
#         clip_num_ = 1
#         video_type_ = VideoType.REST
#     return subject_, video_type_, split_, int(session_num_), int(clip_num_)


def extract_eulerian_signals():
    physiological_parameter_list = ['hr', 'rr']
    for video_file_path in tqdm(video_file_paths):
        subject, video_type, split, session_num = get_details_from_video_path(video_file_path)
        if video_type.name == VideoType.REST.name:
            continue
        frames = extract_frames_from_video(video_file_path)
        for clip_seconds in clip_seconds_list:
            fps = len(frames) / time_duration_for_video_type[video_type]  # CHANGE HERE
            for block_size in block_sizes:
                video_info = {
                    "subject": subject,
                    "label": video_type.name,
                    "clip_seconds": clip_seconds,
                    "block_size": block_size,
                    "split": split,
                    "session_num": session_num,
                }
                for physiological_parameter in physiological_parameter_list:
                    compute_temporal_signals_non_overlapping_clips(frames, destination_path, fps,
                                                                   skip_seconds=SKIP_SECONDS, clip_seconds=clip_seconds,
                                                                   block_size=block_size,
                                                                   video_info=video_info,
                                                                   physiological_parameter=physiological_parameter,
                                                                   display_steps=DISPLAY_INTERMEDIATE_OUTPUTS)


def extract_lagrangian_signals():
    physiological_parameter_list = ['hr', 'rr']
    for video_file_path in tqdm(video_file_paths):
        subject, video_type, split, session_num = get_details_from_video_path(video_file_path)
        if video_type.name == VideoType.REST.name:
            continue
        frames = extract_frames_from_video(video_file_path)
        for clip_seconds in clip_seconds_list:
            fps = len(frames) / time_duration_for_video_type[video_type]  # CHANGE HERE
            video_info = {
                "subject": subject,
                "label": video_type.name,
                "clip_seconds": clip_seconds,
                "split": split,
                "session_num": session_num,
            }
            for physiological_parameter in physiological_parameter_list:
                compute_lagrangian_signals_non_overlapping_clips(frames, destination_path, fps,
                                                                 skip_seconds=SKIP_SECONDS,
                                                                 clip_seconds=clip_seconds,
                                                                 video_info=video_info,
                                                                 physiological_parameter=physiological_parameter,
                                                                 display_steps=DISPLAY_INTERMEDIATE_OUTPUTS)


# def extract_visual_features():
#     for openface_feature_file_path in tqdm(openface_feature_file_paths):
#         subject, video_type, split, session_num = get_details_from_visual_feature_path(openface_feature_file_path)
#         if video_type.name == VideoType.REST.name:
#             continue
#         df_raw = pd.read_csv(openface_feature_file_path)
#         FPS = df_raw.shape[0] / time_duration_for_video_type[video_type]
#         ROWS_TO_SKIP = int(SKIP_SECONDS * FPS)
#         df = df_raw.iloc[ROWS_TO_SKIP:-ROWS_TO_SKIP].reset_index(drop=True)
#         length = df.shape[0]
#
#         for clip_seconds in clip_seconds_list:
#             clip_frame_count = math.floor(DEFAULT_FPS * clip_seconds)
#             clip_count = length // clip_frame_count
#             for clip_number in range(clip_count):
#                 df_clipped = (df.iloc[int(clip_frame_count * clip_number): int(clip_frame_count * (clip_number + 1))].
#                               reset_index(drop=True))
#
#                 clip_info = {
#                     "subject": subject,
#                     "label": video_type.name,
#                     "clip_seconds": clip_seconds,
#                     "split": split,
#                     "session_num": session_num,
#                     "clip_number": clip_number
#                 }
#
#                 save_visual_features(clip_info, df_clipped)
#
#
# def save_visual_features(clip_info, df_clipped):
#     AU_classification = df_clipped.iloc[:, openface_indices.AU_classification]
#     AU_regression = df_clipped.iloc[:, openface_indices.AU_regression]
#     eye_landmark_left_X_3D = df_clipped.iloc[:, openface_indices.eye_landmark_left_X_3D]
#     eye_landmark_left_Y_3D = df_clipped.iloc[:, openface_indices.eye_landmark_left_Y_3D]
#     eye_landmark_left_Z_3D = df_clipped.iloc[:, openface_indices.eye_landmark_left_Z_3D]
#     eye_landmark_right_X_3D = df_clipped.iloc[:, openface_indices.eye_landmark_right_X_3D]
#     eye_landmark_right_Y_3D = df_clipped.iloc[:, openface_indices.eye_landmark_right_Y_3D]
#     eye_landmark_right_Z_3D = df_clipped.iloc[:, openface_indices.eye_landmark_right_Z_3D]
#     eye_landmark_left_x = df_clipped.iloc[:, openface_indices.eye_landmark_left_x]
#     eye_landmark_left_y = df_clipped.iloc[:, openface_indices.eye_landmark_left_y]
#     eye_landmark_right_x = df_clipped.iloc[:, openface_indices.eye_landmark_right_x]
#     eye_landmark_right_y = df_clipped.iloc[:, openface_indices.eye_landmark_right_y]
#     gaze_angles = df_clipped.iloc[:, openface_indices.gaze_angles]
#     gaze_coordinates_left = df_clipped.iloc[:, openface_indices.gaze_coordinates_left]
#     gaze_coordinates_right = df_clipped.iloc[:, openface_indices.gaze_coordinates_right]
#     pose_location = df_clipped.iloc[:, openface_indices.pose_location]
#     pose_rotation = df_clipped.iloc[:, openface_indices.pose_rotation]
#     ###########
#     # PERCLOS #
#     ###########
#     PERCLOS_left_h1 = df_clipped.iloc[:, openface_indices.PERCLOS_left_h1].copy().set_axis(["x", "y"], axis=1)
#     PERCLOS_left_h2 = df_clipped.iloc[:, openface_indices.PERCLOS_left_h2].copy().set_axis(["x", "y"], axis=1)
#     PERCLOS_left_v1 = df_clipped.iloc[:, openface_indices.PERCLOS_left_v1].copy().set_axis(["x", "y"], axis=1)
#     PERCLOS_left_v2 = df_clipped.iloc[:, openface_indices.PERCLOS_left_v2].copy().set_axis(["x", "y"], axis=1)
#     PERCLOS_right_h1 = df_clipped.iloc[:, openface_indices.PERCLOS_right_h1].copy().set_axis(["x", "y"], axis=1)
#     PERCLOS_right_h2 = df_clipped.iloc[:, openface_indices.PERCLOS_right_h2].copy().set_axis(["x", "y"], axis=1)
#     PERCLOS_right_v1 = df_clipped.iloc[:, openface_indices.PERCLOS_right_v1].copy().set_axis(["x", "y"], axis=1)
#     PERCLOS_right_v2 = df_clipped.iloc[:, openface_indices.PERCLOS_right_v2].copy().set_axis(["x", "y"], axis=1)
#     # EAR - Eye Aspect Ratio
#     EAR_values_left = (np.linalg.norm(PERCLOS_left_v1.values - PERCLOS_left_v2.values, axis=1) /
#                        (np.linalg.norm(PERCLOS_left_h1.values - PERCLOS_left_h2.values, axis=1) + epsilon))
#     EAR_values_right = (np.linalg.norm(PERCLOS_right_v1.values - PERCLOS_right_v2.values, axis=1) /
#                         (np.linalg.norm(PERCLOS_right_h1.values - PERCLOS_right_h2.values, axis=1) + epsilon))
#     subject = clip_info["subject"]
#     label = clip_info["label"]
#     clip_seconds = clip_info["clip_seconds"]
#     split = clip_info["split"]
#     session_num = clip_info["session_num"]
#     clip_number = clip_info["clip_number"]
#
#     # Define the directory structure for saving the signals
#     if split is not None:
#         ## This will be triggered if the output folder of 'extract_openface_features.py' is stored split-wise
#         folder_structure = os.path.join(destination_path, "visual_features", f"{clip_seconds:03}s", split, label,
#                                         subject)
#     else:
#         ## This will be triggered if the output folder of 'extract_openface_features.py' is stored at same folder level
#         folder_structure = os.path.join(destination_path, "visual_features", f"{clip_seconds:03}s", label, subject)
#
#     if session_num < 0:
#         file_name = f"{clip_number:02}.npy"
#     else:
#         file_name = f"{session_num:02}_{clip_number:02}.npy"
#
#     name_feature_pair = {
#         "AU_classification": AU_classification,
#         "AU_regression": AU_regression,
#         "eye_landmark_3D": pd.concat([eye_landmark_left_X_3D, eye_landmark_left_Y_3D, eye_landmark_left_Z_3D,
#                                       eye_landmark_right_X_3D, eye_landmark_right_Y_3D, eye_landmark_right_Z_3D], axis=1),
#         "eye_landmark": pd.concat([eye_landmark_left_x, eye_landmark_left_y,
#                                    eye_landmark_right_x, eye_landmark_right_y], axis=1),
#         "gaze": pd.concat([gaze_coordinates_left, gaze_coordinates_right, gaze_angles], axis=1),
#         "pose": pd.concat([pose_location, pose_rotation], axis=1),
#         "EAR_values": np.concatenate((EAR_values_left[:, np.newaxis], EAR_values_right[:, np.newaxis]), axis=1)
#     }
#     for sub_folder, _ in name_feature_pair.items():
#         os.makedirs(os.path.join(folder_structure, sub_folder), exist_ok=True)
#
#     for name, data in name_feature_pair.items():
#         output_path = os.path.join(folder_structure, name, file_name)
#
#         if isinstance(data, pd.DataFrame):
#             np.save(output_path, data.values)
#         else:
#             np.save(output_path, data)


base_path = "/media/user/Projects/Datasets/AI4Pain/2024"
video_source_path_train = f"{base_path}/Train/video/*"
video_source_path_valid = f"{base_path}/Validation/video/*"
destination_path = os.path.join(base_path, "Extracted_Output")

video_file_paths = natsorted(glob(os.path.join(video_source_path_train, "*.mp4")) +
                             glob(os.path.join(video_source_path_valid, "*.mp4")))
openface_feature_file_paths = natsorted(glob(os.path.join(f"{base_path}/Extracted_OF_5/*/*", "*.csv")))

clip_seconds_list = [3, 4, 9]

block_sizes = [41]  # Note should only be used if adaptive block size is False
if USE_ADAPTIVE_BLOCK_SIZE:
    assert len(block_sizes) == 1

extract_eulerian_signals()
extract_lagrangian_signals()
# extract_visual_features()
