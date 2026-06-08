import os
from glob import glob
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset


import BSS
from BSS import get_rppg_from_temporal_signals, compute_snr_for_signal
from extract_temporal_signal_files import VideoType, get_details_from_rppg_path, DEFAULT_FPS
from scripts.utils import SPLIT_USED

data_folder = "/media/user/Projects/Datasets/AI4Pain/2024/Extracted"

all_visual_feature_types = [
    "AU_classification",
    "AU_regression",
    "eye_landmark_3D",
    "eye_landmark",
    "gaze",
    "pose",
    "EAR_values"
]

list_of_all_colors = ['avg', 'blue', 'chrom', 'green', 'red', 'w_avg']


class AI4PainDataset(Dataset):
    """
    A PyTorch Dataset class to load and serve rPPG signals and visual features for the AI4Pain 2024 dataset.

    This dataset supports:
    - Dynamic selection of rPPG feature types based on color channels (e.g., green, red, chrominance).
    - Extraction of physiological parameters: heart rate (HR), respiratory rate (RR), or both.
    - Visual features such as Action Units (AUs), gaze, pose, and eye landmarks.
    - Per-clip loading with configurable clip durations, block sizes, and BSS parameters.

    Attributes:
        feature_type (str): Type of features to load - 'rppg', 'visual', or 'both'.
        physiological_parameter (str): Physiological parameter to compute - 'hr', 'rr', or 'both'.
        clip_seconds (str): Clip duration in seconds, formatted as a string (e.g., "009s").
        block_size (str): Block size used during preprocessing, formatted as a string (e.g., "011bs").
        device (str): Device to place the tensors on ('cuda' or 'cpu').
        rppg_feature_color_list (list): List of color-based signals to use (e.g., ['green', 'chrom']).
        visual_feature_type_list (list): List of visual features to load (e.g., ['gaze', 'pose']).
        bss_algorithm (str): BSS algorithm used to extract rPPG signals - 'muk', 'pca', or 'ica'.
        top_k_per (int): Top-k percentage for selecting patches based on quality.
        quality_parameter (str): Signal quality measure used for patch ranking -'snr', 'sigma', or 'both'.
    """

    def __init__(self, feature_type="both", physiological_parameter="both", clip_seconds=9, block_size=11,
                 split="Train", label=VideoType.BASELINE.name, subject="*", rppg_feature_color_list=None,
                 visual_feature_type_list=None, device="cuda", bss_algorithm="muk", top_k_per=10,
                 quality_parameter="both", au_list=None):
        """
           Initializes the RPPGDataset for loading rPPG signals and visual features.

           Args:
               feature_type (str): One of {'rppg', 'visual', 'both'}, indicating which features to return.
               physiological_parameter (str): One of {'hr', 'rr', 'both'}, specifying the target physiological signal.
               clip_seconds (int or str): Duration of each video clip, e.g., 9 or '009s'.
               block_size (int or str): Patch/block size used for region-based signal extraction.
               split (str): Dataset split, e.g., 'Train', 'Validation', or 'Test'.
               label (str): Video type label (e.g., 'BASELINE', 'LOW_PAIN', 'HIGH_PAIN').
               subject (str): Subject ID or '*' to include all subjects.
               rppg_feature_color_list (list or None): List of color channels to use (e.g., ['green', 'chrom']).
               visual_feature_type_list (list or None): List of visual features to load (e.g., ['gaze', 'pose']).
               device (str): Target device to load tensors on, 'cuda' or 'cpu'.
               bss_algorithm (str): Blind source separation algorithm; one of {'muk', 'pca', 'ica'}.
               top_k_per (int): Percentage of top-quality patches to select (0 < top_k_per ≤ 100).
               quality_parameter (str): Signal quality metric; one of {'snr', 'sigma', 'both'}.
               au_list(list or None): List of indices of AUs to use.
        """
        super().__init__()

        assert feature_type in ["rppg", "visual", "both"], f"Invalid feature_type: {feature_type}"
        assert physiological_parameter in ["hr", "rr",
                                           "both"], f"Invalid physiological_parameter: {physiological_parameter}"
        assert isinstance(top_k_per, int) and (0 < top_k_per <= 100), "top_k_per must be an integer in (0, 100]"
        assert quality_parameter in ["snr", "sigma", "both"], f"Invalid quality_parameter: {quality_parameter}"
        assert bss_algorithm in ["muk", "pca", "ica"], f"Invalid bss_algorithm: {bss_algorithm}"
        assert device in ["cuda", "cpu"], f"Invalid device: {device}"

        self.data_folder = data_folder
        self.feature_type = feature_type
        self.physiological_parameter = physiological_parameter
        self.clip_seconds = f"{clip_seconds:03}s" if type(clip_seconds) is int else clip_seconds
        if isinstance(block_size, int):
            self.block_size = f"{block_size:03}bs"
        elif block_size == "adaptive":
            self.block_size = "adaptive_bs"
        else:
            self.block_size = block_size
        self.subject = str(subject)
        self.split = split
        self.label = label
        self.bss_algorithm = bss_algorithm
        self.top_k_per = top_k_per
        self.quality_parameter = quality_parameter
        self.all_colors = False
        self.au_list = au_list

        if visual_feature_type_list is not None:
            self.visual_feature_type_list = visual_feature_type_list
        else:
            self.visual_feature_type_list = all_visual_feature_types

        if rppg_feature_color_list is not None:
            self.rppg_feature_color_list = rppg_feature_color_list
        else:
            self.rppg_feature_color_list = list_of_all_colors
            self.all_colors = True

        self.device = device

        self.rppg_feature_color_files_mapping = {}

        self._initialize_rppg_feature_paths()

        self._initialize_visual_feature_paths()

        self._validate_file_lengths()

    def _initialize_rppg_feature_paths(self):
        for color in self.rppg_feature_color_list:
            # self.rppg_feature_color_files_mapping[color] = {
            #     "hr": natsorted(glob(os.path.join(self.data_folder, "rppg_signals", "hr",
            #                                       self.clip_seconds, self.block_size, self.split, self.label,
            #                                       self.subject, color,
            #                                       "*.npy"))),
            #     "rr": natsorted(glob(os.path.join(self.data_folder, "rppg_signals", "rr",
            #                                       self.clip_seconds, self.block_size, self.split, self.label,
            #                                       self.subject, color,
            #                                       "*.npy")))
            # }
            hr_rr_mapping = {}
            for param in ["hr", "rr"]:
                if SPLIT_USED:
                    path = os.path.join(self.data_folder, "rppg_signals", param,
                                        self.clip_seconds, self.block_size, self.split, self.label,
                                        self.subject, color, "*.npy")
                else:
                    path = os.path.join(self.data_folder, "rppg_signals", param,
                                        self.clip_seconds, self.block_size, self.label,
                                        self.subject, color, "*.npy")
                hr_rr_mapping[param] = natsorted(glob(path))
            self.rppg_feature_color_files_mapping[color] = hr_rr_mapping

    def _initialize_visual_feature_paths(self):
        # au_classification_folder_structure = os.path.join(self.data_folder,
        #                                                   'visual_features',
        #                                                   self.clip_seconds, self.split, self.label,
        #                                                   self.subject, "AU_classification", "*.npy")
        # au_regression_folder_structure = os.path.join(self.data_folder,
        #                                               'visual_features',
        #                                               self.clip_seconds, self.split, self.label,
        #                                               self.subject, "AU_regression", "*.npy")
        # eye_landmark_3D_folder_structure = os.path.join(self.data_folder,
        #                                                 'visual_features',
        #                                                 self.clip_seconds, self.split, self.label,
        #                                                 self.subject, "eye_landmark_3D", "*.npy")
        # eye_landmark_folder_structure = os.path.join(self.data_folder,
        #                                              'visual_features',
        #                                              self.clip_seconds, self.split, self.label,
        #                                              self.subject, "eye_landmark", "*.npy")
        # gaze_folder_structure = os.path.join(self.data_folder,
        #                                      'visual_features',
        #                                      self.clip_seconds, self.split, self.label,
        #                                      self.subject, "gaze", "*.npy")
        # pose_folder_structure = os.path.join(self.data_folder,
        #                                      'visual_features',
        #                                      self.clip_seconds, self.split, self.label,
        #                                      self.subject, "pose", "*.npy")
        # EAR_values_folder_structure = os.path.join(self.data_folder,
        #                                            'visual_features',
        #                                            self.clip_seconds, self.split, self.label,
        #                                            self.subject, "EAR_values", "*.npy")
        # self.au_classification_files = natsorted(glob(au_classification_folder_structure))
        # self.au_regression_files = natsorted(glob(au_regression_folder_structure))
        # self.eye_landmark_3D_files = natsorted(glob(eye_landmark_3D_folder_structure))
        # self.eye_landmark_files = natsorted(glob(eye_landmark_folder_structure))
        # self.gaze_files = natsorted(glob(gaze_folder_structure))
        # self.pose_files = natsorted(glob(pose_folder_structure))
        # self.EAR_values_files = natsorted(glob(EAR_values_folder_structure))

        if SPLIT_USED:
            vis_base = os.path.join(self.data_folder, 'visual_features',
                                    self.clip_seconds, self.split, self.label, self.subject)
        else:
            vis_base = os.path.join(self.data_folder, 'visual_features',
                                    self.clip_seconds, self.label, self.subject)

        self.au_classification_files = natsorted(glob(os.path.join(vis_base, "AU_classification", "*.npy")))
        self.au_regression_files = natsorted(glob(os.path.join(vis_base, "AU_regression", "*.npy")))
        self.eye_landmark_3D_files = natsorted(glob(os.path.join(vis_base, "eye_landmark_3D", "*.npy")))
        self.eye_landmark_files = natsorted(glob(os.path.join(vis_base, "eye_landmark", "*.npy")))
        self.gaze_files = natsorted(glob(os.path.join(vis_base, "gaze", "*.npy")))
        self.pose_files = natsorted(glob(os.path.join(vis_base, "pose", "*.npy")))
        self.EAR_values_files = natsorted(glob(os.path.join(vis_base, "EAR_values", "*.npy")))

    def _validate_file_lengths(self):
        if self.feature_type == "rppg" or self.feature_type == "both":
            # Validate that all colors have the same number of HR and RR files
            first_color = self.rppg_feature_color_list[0]
            expected_len = len(self.rppg_feature_color_files_mapping[first_color]['hr'])
            for color in self.rppg_feature_color_list:
                hr_files = self.rppg_feature_color_files_mapping[color]['hr']
                rr_files = self.rppg_feature_color_files_mapping[color]['rr']
                assert len(hr_files) == expected_len, f"Mismatch in HR files for color: {color}"
                assert len(rr_files) == expected_len, f"Mismatch in RR files for color: {color}"

        if self.feature_type == "visual" or self.feature_type == "both":
            # Validate that all visual feature types have the same number of files
            assert (len(self.au_classification_files) ==
                    len(self.au_regression_files) == len(self.eye_landmark_3D_files) == len(self.eye_landmark_files) ==
                    len(self.gaze_files) == len(self.pose_files) == len(self.EAR_values_files)), \
                "Mismatch in the number of files across visual feature types."

        if self.feature_type == "both":
            # Cross-check rppg and visual file counts
            for color in self.rppg_feature_color_list:
                hr_files = self.rppg_feature_color_files_mapping[color]['hr']
                rr_files = self.rppg_feature_color_files_mapping[color]['rr']
                assert len(hr_files) == len(self.au_classification_files), \
                    f"Mismatch between HR files and visual features for color: {color}"
                assert len(rr_files) == len(self.au_classification_files), \
                    f"Mismatch between RR files and visual features for color: {color}"

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        if self.feature_type == "rppg":
            first_color = self.rppg_feature_color_list[0]
            return len(self.rppg_feature_color_files_mapping[first_color]['hr'])
        return len(self.au_classification_files)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            sample (dict): A dictionary containing the visual features and rPPG signals.
        """
        rppg_features_dict = []
        visual_feature_dict = []

        if self.feature_type == "rppg" or self.feature_type == "both":
            rppg_features_dict = self.extract_rppg_features(idx)
        if self.feature_type == "visual" or self.feature_type == "both":
            visual_feature_dict = self.extract_visual_features(idx)

        if self.feature_type == "visual":
            file_path = self.au_classification_files[idx]
        else:
            file_path = self.rppg_feature_color_files_mapping[self.rppg_feature_color_list[0]]['hr'][idx]

        subject, video_type, split, session_number, clip_number = get_details_from_rppg_path(file_path)
        gt_label = video_type.value
        return {"rppg_features_dict": rppg_features_dict,
                "visual_feature_dict": visual_feature_dict,
                "subject": subject,
                "split": split,
                "session_number": session_number,
                "clip_number": clip_number,
                }, gt_label

    def extract_rppg_features(self, index):
        """
        Extracts rPPG features for a given sample index across all specified color channels.

        For each color in `self.rppg_feature_color_list`, this method retrieves the corresponding HR and RR signal
        file paths and calls `extract_rppg_features_for_single_color()` to compute the final features.

        Args:
            index (int): Index of the sample to process.

        Returns:
            dict: A dictionary where each key is a color (e.g., 'green', 'chrom') and the value is a
                  torch.Tensor containing the processed rPPG features for that color.
                  The shape depends on whether `physiological_parameter` is 'hr', 'rr', or 'both'.
        """
        rppg_features_dict = {key: torch.tensor([]) for key in self.rppg_feature_color_list}

        for color_type, _ in rppg_features_dict.items():
            files = self.rppg_feature_color_files_mapping[color_type]
            rppg_features_dict[color_type] = self.extract_rppg_features_for_single_color(index, files["hr"],
                                                                                         files["rr"])
        return rppg_features_dict

    def extract_rppg_features_for_single_color(self, index, rppg_hr_files, rppg_rr_files):
        """
        Extracts rPPG features from HR and/or RR signals for a specific color channel at the given index.

        Depending on the value of `self.physiological_parameter`, this method:
        - Loads HR signals and processes them if 'hr'
        - Loads RR signals and processes them if 'rr'
        - Loads and processes both if 'both', and stacks them along the last axis

        The method uses the configured BSS algorithm, quality metric, and sampling rate during processing.

        Args:
            index (int): Index of the sample to process.
            rppg_hr_files (list): List of HR signal file paths for the specific color.
            rppg_rr_files (list): List of RR signal file paths for the specific color.

        Returns:
            torch.Tensor: A tensor containing the extracted rPPG features. Shape depends on the physiological parameter:
                          - (1, F), if 'hr' or 'rr'
                          - (2, F), if 'both' (where F = number of features per signal)
        """
        rppg_features = None

        if self.physiological_parameter == 'hr':
            signals = np.load(rppg_hr_files[index])
            rppg_features = torch.from_numpy(
                get_rppg_from_temporal_signals(signals, sampling_rate=DEFAULT_FPS, bss_algorithm=self.bss_algorithm,
                                               top_k_per=self.top_k_per,
                                               quality_parameter=self.quality_parameter,
                                               physiological_parameter=self.physiological_parameter).copy())
        elif self.physiological_parameter == 'rr':
            signals = np.load(rppg_rr_files[index])
            rppg_features = torch.from_numpy(
                get_rppg_from_temporal_signals(signals, sampling_rate=DEFAULT_FPS, bss_algorithm=self.bss_algorithm,
                                               top_k_per=self.top_k_per,
                                               quality_parameter=self.quality_parameter,
                                               physiological_parameter=self.physiological_parameter).copy())
        elif self.physiological_parameter == "both":
            hr_signals = np.load(rppg_hr_files[index])
            hr_rppg = get_rppg_from_temporal_signals(hr_signals, sampling_rate=DEFAULT_FPS,
                                                     bss_algorithm=self.bss_algorithm,
                                                     top_k_per=self.top_k_per,
                                                     quality_parameter=self.quality_parameter,
                                                     physiological_parameter='hr')
            rr_signals = np.load(rppg_rr_files[index])
            rr_rppg = get_rppg_from_temporal_signals(rr_signals, sampling_rate=DEFAULT_FPS,
                                                     bss_algorithm=self.bss_algorithm,
                                                     top_k_per=self.top_k_per,
                                                     quality_parameter=self.quality_parameter,
                                                     physiological_parameter='rr')
            rppg_features = torch.from_numpy(np.row_stack([hr_rppg, rr_rppg]).copy())
        return rppg_features

    def extract_visual_features(self, index):
        """
        Loads and returns visual features for the given sample index.

        For each feature type listed in `self.visual_feature_type_list`, this method:
        - Loads the corresponding `.npy` file.
        - Converts it into a PyTorch tensor.
        - Adds it to a dictionary keyed by the feature type.

        Args:
            index (int): Index of the sample to load visual features for.

        Returns:
            dict: A dictionary containing visual features as PyTorch tensors.
                  Keys correspond to feature types (e.g., 'gaze', 'pose', 'AU_classification').
        """
        visual_feature_dict = {key: torch.tensor([]) for key in self.visual_feature_type_list}
        for visual_feature_type, _ in visual_feature_dict.items():
            if visual_feature_type == "AU_classification":
                au_classification_data = np.load(self.au_classification_files[index])
                if self.au_list:
                    au_classification_data = au_classification_data[:, self.au_list]
                visual_feature_dict[visual_feature_type] = torch.from_numpy(au_classification_data).T
            elif visual_feature_type == "AU_regression":
                au_regression_data = np.load(self.au_regression_files[index])
                if self.au_list:
                    au_regression_data = au_regression_data[:, self.au_list]
                visual_feature_dict[visual_feature_type] = torch.from_numpy(au_regression_data).T
            elif visual_feature_type == "eye_landmark_3D":
                visual_feature_dict[visual_feature_type] = torch.from_numpy(
                    np.load(self.eye_landmark_3D_files[index])).T
            elif visual_feature_type == "eye_landmark":
                visual_feature_dict[visual_feature_type] = torch.from_numpy(np.load(self.eye_landmark_files[index])).T
            elif visual_feature_type == "gaze":
                visual_feature_dict[visual_feature_type] = torch.from_numpy(np.load(self.gaze_files[index])).T
            elif visual_feature_type == "pose":
                visual_feature_dict[visual_feature_type] = torch.from_numpy(np.load(self.pose_files[index])).T
            elif visual_feature_type == "EAR_values":
                visual_feature_dict[visual_feature_type] = torch.from_numpy(np.load(self.EAR_values_files[index])).T
        return visual_feature_dict

# dataset = RPPGDataset()
# for i in range(len(dataset)):
#     data = dataset[i]
#     print(data)
