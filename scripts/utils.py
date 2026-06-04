import numpy as np
from scipy.signal import butter, filtfilt

SPLIT_USED = False

SUBJECT_LIST_TRAIN = []
SUBJECT_LIST_TEST = []
SUBJECT_LIST_VALIDATION = []

if SPLIT_USED:
    subject_lists = {
        "train": SUBJECT_LIST_TRAIN,
        "test": SUBJECT_LIST_TEST,
        "validation": SUBJECT_LIST_VALIDATION,
    }

    for split_name, subjects in subject_lists.items():
        assert len(subjects) > 0, f"{split_name} subject list must not be empty when SPLIT_USED=True"

    train_subjects = set(SUBJECT_LIST_TRAIN)
    test_subjects = set(SUBJECT_LIST_TEST)
    validation_subjects = set(SUBJECT_LIST_VALIDATION)

    assert train_subjects.isdisjoint(test_subjects), "Train and test subjects overlap"
    assert train_subjects.isdisjoint(validation_subjects), "Train and validation subjects overlap"
    assert test_subjects.isdisjoint(validation_subjects), "Test and validation subjects overlap"


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


openface_indices = DotDict({
    "AU_classification": [656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673],
    "AU_regression": [639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655],
    "confidence": [3],
    "eye_landmark_left_X_3D": [125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                               142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152],
    "eye_landmark_left_Y_3D": [181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
                               198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208],
    "eye_landmark_left_Z_3D": [265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
                               282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292],
    "eye_landmark_left_x": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                            35, 36, 37, 38, 39, 40],
    "eye_landmark_left_y": [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                            91, 92, 93, 94, 95, 96],
    "eye_landmark_right_X_3D": [153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
                                170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180],
    "eye_landmark_right_Y_3D": [209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
                                226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236],
    "eye_landmark_right_Z_3D": [237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
                                254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264],
    "eye_landmark_right_x": [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                             63, 64, 65, 66, 67, 68],
    "eye_landmark_right_y": [97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                             115, 116, 117, 118, 119, 120, 121, 122, 123, 124],
    "face_id": [1],
    "frame_number": [0],
    "gaze_angles": [11, 12],
    "gaze_coordinates_left": [5, 6, 7],
    "gaze_coordinates_right": [8, 9, 10],
    "landmark_X_3D": [435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453,
                      454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472,
                      473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491,
                      492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502],
    "landmark_Y_3D": [503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521,
                      522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540,
                      541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
                      560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570],
    "landmark_Z_3D": [571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589,
                      590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608,
                      609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627,
                      628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638],
    "landmark_x": [299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
                   318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,
                   337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355,
                   356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366],
    "landmark_y": [367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385,
                   386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,
                   405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423,
                   424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434],
    "pose_location": [293, 294, 295],
    "pose_rotation": [296, 297, 298],
    "success": [4],
    "timestamp": [2],

    'AU01': [639, 656],
    "AU02": [640, 657],
    "AU04": [641, 658],
    "AU05": [642, 659],
    "AU06": [643, 660],
    "AU07": [644, 661],
    "AU09": [645, 662],
    "AU10": [646, 663],
    "AU12": [647, 664],
    "AU14": [648, 665],
    "AU15": [649, 666],
    "AU17": [650, 667],
    "AU20": [651, 668],
    "AU23": [652, 669],
    "AU25": [653, 670],
    "AU26": [654, 671],
    "AU28": [655, 672],
    "AU45": [673],

    "PERCLOS_left_h1": [21, 77],
    "PERCLOS_left_h2": [27, 83],
    "PERCLOS_left_v1": [24, 80],
    "PERCLOS_left_v2": [30, 86],
    "PERCLOS_right_h1": [49, 105],
    "PERCLOS_right_h2": [55, 111],
    "PERCLOS_right_v1": [52, 108],
    "PERCLOS_right_v2": [58, 114],

})

epsilon = 1e-8


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
