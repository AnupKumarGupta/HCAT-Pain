import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA

from utils import get_lowcut_highcut_frequencies_based_on_physiological_parameter, butter_bandpass_filter


def plot_signal(signal, x_axis=None, title_text=None):
    """
    Plots a given signal with an optional x-axis.

    Inputs:
    -------
    signal : list or array-like
        The data points of the signal to be plotted.
    x_axis : list or array-like, optional
        The values for the x-axis, if provided. If None, the signal will be plotted against its index.
    title_text : string, optional
        The text to be displayed on the plot.

    Outputs:
    -------
    None
        This function displays a plot but does not return any value.
    """

    if x_axis is None:
        plt.plot(signal)
    else:
        plt.plot(x_axis, signal)

    if title_text:
        plt.title(title_text)
    plt.show()


def plot_signals(signals, x_axis=None, labels=None, colors=None):
    """
    Plots multiple signals on the same plot with optional x-axis values, labels, and colors.

    Inputs:
    -------
    signals : list or array-like of 1D arrays
        A list containing multiple signals to be plotted.
    x_axis : array-like, optional
        Common x-axis values for all signals. If None, signal indices will be used.
    labels : list of str, optional
        Labels for each signal. If None, no legend is displayed unless colors are used.
    colors : list of str or tuples, optional
        Colors for each signal. If None, random distinct colors will be used.

    Outputs:
    --------
    None
        This function displays a plot but does not return any value.
    """
    num_signals = len(signals)

    # Generate random non-repeating colors if not provided
    if colors is None:
        colors = []
        for _ in range(num_signals):
            color = (random.random(), random.random(), random.random())
            while color in colors:  # Ensure uniqueness
                color = (random.random(), random.random(), random.random())
            colors.append(color)

    # Plot each signal
    for idx, signal in enumerate(signals):
        label = labels[idx] if labels and idx < len(labels) else None
        color = colors[idx]
        if x_axis is None:
            plt.plot(signal, label=label, color=color)
        else:
            plt.plot(x_axis, signal, label=label, color=color)

    if labels:
        plt.legend()

    plt.tight_layout()
    plt.show()


def MUK_algorithm(received_signal, nt, mu=2e-2, kurt_sign=-1):
    """
    Blind Source Separation based on Multiuser Kurtosis Maximization.

    Inputs:
    -------
    received_signal : np.ndarray
        The received signal matrix (nr x N), where nr is the number of received signals, and N is the signal length.
    nt : int
        The number of sources to estimate (should be less than nr).
    mu : float, optional
        The step size for gradient descent (default is 2e-2).
    kurt_sign : int, optional
        The sign of the kurtosis for maximization or minimization (default is -1).

    Outputs:
    --------
    estimated_sources : np.ndarray
        The estimated source signals (nt x N matrix).
    estimated_channel : np.ndarray
        The estimated channel matrix (nr x nt).
    """
    # Dimensions of the received signal matrix
    nr, length_signal = received_signal.shape
    received_signal = np.asarray(received_signal, dtype=np.float64)

    # Step 1: Perform Principal Component Analysis (PCA) for dimensionality reduction
    received_signal = received_signal - np.mean(received_signal, axis=1, keepdims=True)
    R = np.dot(received_signal, received_signal.T) / length_signal  # Compute covariance matrix
    V, U = np.linalg.eig(R)  # Eigen decomposition of covariance matrix
    index_sorted = np.argsort(V)[::-1]  # Sort eigenvalues in descending order
    V_sorted = V[index_sorted]
    V_reduced = np.diag(V_sorted[:nt])  # Retain top nt eigenvalues
    U_reduced = U[:, index_sorted[:nt]]  # Retain corresponding eigenvectors

    # Whiten the data using PCA components
    inverse_sqrt_V_reduced = np.linalg.inv(np.sqrt(V_reduced))
    U_reduced_transpose = U_reduced.T
    whitened_data = np.dot(np.dot(inverse_sqrt_V_reduced, U_reduced_transpose), received_signal)

    # Step 2: Perform Kurtosis Maximization using gradient ascent
    W = np.eye(nt)  # Initialize separation matrix
    for k in range(length_signal):
        Yk = whitened_data[:, k]  # Extract k-th column (current signal snapshot)
        z_k = W.T @ Yk  # Source estimate at current time

        # Calculate gradient term for kurtosis maximization
        Z_k = ((np.abs(z_k) ** 2) * z_k).T

        # Update W using gradient descent
        W_prime = W + mu * kurt_sign * np.conj(Yk) * Z_k

        # Enforce Unitary Constraint (Gram-Schmidt Orthogonalization)
        W_new = np.zeros_like(W_prime)
        W_new[:, 0] = W_prime[:, 0] / np.sqrt(np.sum(np.abs(W_prime[:, 0]) ** 2))

        # Orthogonalize columns iteratively if nt > 1
        for column_index in range(1, nt):
            first_term = np.dot(W_new[:, :column_index].conj().T, W_prime[:, column_index])
            numerator = W_prime[:, column_index] - np.dot(W_new[:, :column_index], first_term)
            denominator = np.sqrt(np.sum(np.abs(numerator) ** 2))
            W_new[:, column_index] = numerator / denominator

        # Update W with the orthogonalized matrix
        W = W_new

    # Calculate the estimated sources and channel matrix
    estimated_sources = np.dot(W.T, whitened_data)
    estimated_channel = np.dot(U_reduced, np.dot(np.sqrt(np.diag(V_reduced)), np.conj(W)))

    # Ensure outputs are real if inputs were real
    estimated_sources = estimated_sources.real if np.iscomplexobj(estimated_sources) else estimated_sources
    estimated_channel = estimated_channel.real if np.iscomplexobj(estimated_channel) else estimated_channel

    return estimated_sources, estimated_channel


def get_rppg_from_temporal_signals(signals, sampling_rate=35, bss_algorithm="muk", top_k_per=1,
                                   quality_parameter="snr", physiological_parameter='hr'):
    """
    Extracts the remote photoplethysmography (rPPG) signal from a set of temporal signals by applying
    Blind Source Separation (BSS) techniques such as MUK, PCA, or ICA.

    Inputs:
    -------
    signals : np.ndarray
        A 2D array of temporal signals, where rows represent Regions of Interest (ROIs) and columns represent frames.
    sampling_rate : int, optional
        Sampling rate of the signals in Hz (default is 35 Hz).
    bss_algorithm : str, optional
        The BSS algorithm to use for extracting the rPPG signal. Choices are 'muk', 'pca', or 'ica' (default is 'muk').
    top_k_per : float, optional
        The percentage of ROIs to select based on the quality parameter (default is 1).
    quality_parameter : str, optional
        The quality metric to select ROIs. Choices are 'sigma', 'snr', 'both', or 'none' (default is 'snr').

    Outputs:
    --------
    rppg : np.ndarray
        The extracted rPPG signal as a 1D array.
    """

    assert quality_parameter in ["sigma", "snr", "both", "none"], ("quality_parameter must be either 'sigma', 'snr', "
                                                                   "'both' or 'none'")
    assert bss_algorithm in ["muk", "pca", "ica"], "bss_algorithm must be either 'muk', 'pca' or 'ica'"
    assert 0 < top_k_per <= 100, "top_k_per must belong to (0, 100]"

    rois, frames = signals.shape
    top_k = max(1, int((top_k_per / 100) * rois))

    if quality_parameter == "sigma" or quality_parameter == "snr":
        top_k_signals = extract_top_k_signals(signals, top_k, quality_parameter, sampling_rate, physiological_parameter)
    elif quality_parameter == "both":
        # First filter by sigma to obtain top 2k signals, then filter top k signals by SNR for improved quality
        # selection
        top_2k_signals = extract_top_k_signals(signals, top_k * 2, "sigma", sampling_rate, physiological_parameter)
        top_k_signals = extract_top_k_signals(top_2k_signals, top_k, "snr", sampling_rate, physiological_parameter)
    else:
        # No quality filtering, use all signals
        top_k_signals = signals

    if bss_algorithm == "muk":
        rppg = MUK_algorithm(top_k_signals, nt=1)[0][0]
    elif bss_algorithm == "pca":
        pca = PCA(n_components=1)
        rppg = pca.fit_transform(top_k_signals.T)[:, 0]
    elif bss_algorithm == "ica":
        ica = FastICA(n_components=1)
        rppg = ica.fit_transform(top_k_signals.T)[:, 0]
    else:
        ValueError("Incorrect bss_algorithm value; must be 'sigma' or 'snr'.")
        exit(-1)

    lowcut, highcut = get_lowcut_highcut_frequencies_based_on_physiological_parameter(physiological_parameter)
    rppg = butter_bandpass_filter(rppg, lowcut, highcut, fs=sampling_rate)[0]

    return rppg


def extract_top_k_signals(signals, top_k, quality_parameter, sampling_rate=35, physiological_parameter='hr', main_band=15):
    """
    Selects the top-K signals based on a specified quality metric, either signal standard deviation ('sigma')
    or signal-to-noise ratio ('snr') within a specific heart rate frequency band. This function aims to isolate
    signals with the highest quality for further analysis, such as rPPG extraction.

    Inputs:
    -------
    signals : np.ndarray
        A 2D array where each row represents a Region of Interest (ROI) signal over time.
    top_k : int
        The number of signals to select based on the quality metric.
    quality_parameter : str
        The quality metric used for signal selection, either 'sigma' for standard deviation or 'snr' for
        signal-to-noise ratio.
    sampling_rate : int, optional
        Sampling rate of the signals in Hz (default is 35 Hz).
    main_band : int, optional
        The width of the frequency band (in terms of frequency bins) centered around the peak to be used for
        signal power calculation (default is 5).

    Outputs:
    --------
    selected_signals : np.ndarray
        A subset of the original signals containing the top-K selected signals based on the specified quality metric.

    Raises:
    -------
    ValueError
        If the quality parameter is not 'sigma' or 'snr'.
    """
    # Set FFT size based on the number of frames in each signal
    # fft_size = signals.shape[1]
    fft_size = 12000

    if quality_parameter == "sigma":
        # Quality assessment based on signal standard deviation (sigma)
        sigma = np.std(signals, axis=1)
        qp_sigma = 1.0 / (sigma + 1e-8)
        sorted_indices = np.argsort(qp_sigma)[::-1]  # Sort indices in descending order of quality
        selected_signals = signals[sorted_indices[:top_k]]

    elif quality_parameter == "snr":
        # Quality assessment based on Signal-to-Noise Ratio (SNR)
        fft_signals = np.fft.fft(signals, n=fft_size, axis=1)  # Compute FFT of each ROI signal
        power_spectrums = np.abs(fft_signals) ** 2  # Calculate power spectrum for each signal
        freqs = np.fft.fftfreq(fft_size, 1 / sampling_rate)  # Frequency values for FFT bins

        min_freq, max_freq = get_lowcut_highcut_frequencies_based_on_physiological_parameter(physiological_parameter)

        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)  # Mask for frequencies within HR range
        filtered_power_spectrum = np.where(freq_mask, power_spectrums, 0)  # Zero out frequencies outside HR range

        # Find the peak index within the HR band for each signal
        peak_indices = np.argmax(filtered_power_spectrum[:, freq_mask], axis=1)

        # Adjust peak indices to align with the full power spectrum:
        # The indices in 'peak_indices' represent the highest power frequencies within the limited HR range.
        # However, these indices are relative to the HR band, not the full power spectrum.
        # To align these indices with the original power spectrum:
        # 1. Use `np.flatnonzero(freq_mask)` to find the indices where the frequencies in `freqs` fall
        # within the HR band.
        # 2. The first element, `np.flatnonzero(freq_mask)[0]`, gives the starting index of the HR band within
        # the full power spectrum.
        # 3. Adding this starting index to `peak_indices` adjusts each peak to its correct position
        # in the full spectrum.
        adjusted_peak_indices = np.flatnonzero(freq_mask)[0] + peak_indices

        # Calculate signal power and noise power for SNR calculation
        signal_powers = []
        noise_powers = []

        for idx, peak_idx in enumerate(adjusted_peak_indices):
            # Ensure peak_idx is within bounds for main_band selection
            start_idx = max(0, peak_idx - (main_band - 1) // 2)
            end_idx = min(fft_size, peak_idx + (main_band - 1) // 2 + 1)

            # Create a mask for the narrow main band around the dominant peak
            main_band_mask = np.zeros_like(freq_mask, dtype=bool)
            main_band_mask[start_idx:end_idx] = True

            # Ensure the main band remains inside the physiological frequency range
            main_band_mask = main_band_mask & freq_mask

            # Signal power: power around the dominant frequency inside the physiological band
            signal_power = np.sum(power_spectrums[idx, main_band_mask])
            signal_powers.append(signal_power)

            # Band power: total power inside the physiological frequency range
            band_power = np.sum(power_spectrums[idx, freq_mask])

            # Noise power: remaining physiological-band power after excluding the main band
            noise_power = band_power - signal_power
            noise_powers.append(noise_power)

        # Calculate SNR for each signal
        snrs = np.true_divide(signal_powers, np.array(noise_powers) + 1e-8)
        sorted_indices = np.argsort(snrs)[::-1]
        topk_indices = sorted_indices[:top_k]
        selected_signals = signals[topk_indices]
    else:
        # Raise an error if quality parameter is invalid
        raise ValueError("Incorrect quality parameter; must be 'sigma' or 'snr'.")

    return selected_signals


def compute_snr_for_signal(signal, sampling_rate=35, main_band=15, physiological_parameter='hr', display=False,
                           return_spectrum=False):
    signal = signal - np.mean(signal)
    fft_size = 12000
    fft_signal = np.fft.rfft(signal, n=fft_size)  # Compute FFT of each ROI signal
    power_spectrum = np.abs(fft_signal) ** 2  # Calculate power spectrum for each signal
    freqs = np.fft.rfftfreq(fft_size, 1 / sampling_rate)  # Frequency values for FFT bins
    min_freq, max_freq = get_lowcut_highcut_frequencies_based_on_physiological_parameter(physiological_parameter)
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)  # Mask for frequencies within HR range
    filtered_power_spectrum = np.where(freq_mask, power_spectrum, 0)  # Zero out frequencies outside HR range
    peak_index = np.argmax(filtered_power_spectrum[freq_mask])
    adjusted_peak_index = np.flatnonzero(freq_mask)[0] + peak_index
    start_idx = max(0, adjusted_peak_index - (main_band - 1) // 2)
    end_idx = min(fft_size, adjusted_peak_index + (main_band - 1) // 2 + 1)
    signal_power = np.sum(power_spectrum[start_idx:end_idx])
    total_power = np.sum(power_spectrum)
    noise_power = total_power - signal_power
    snr = np.true_divide(signal_power, np.array(noise_power) + 1e-8)
    hr = freqs[adjusted_peak_index] * 60
    if display:
        plot_signal(power_spectrum, freqs*60)
    if return_spectrum:
        return snr, hr, power_spectrum, freqs*60
    return snr, hr
