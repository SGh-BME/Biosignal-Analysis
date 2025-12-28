import numpy as np
import os
import pandas as pd

def compute_bispectrum(signal, fs=256, nfft=128, window_type='hann', noverlap=64, directory=None):
    """
    Computes the bispectrum of a 1D signal using overlapping windowed FFT segments.

    Parameters:
        signal (1D np.array): Input time-domain signal
        fs (int): Sampling frequency
        nfft (int): FFT size
        window_type (str): Type of window to use ('hann', 'rectangular', etc.)
        noverlap (int): Number of samples to overlap between segments

    Returns:
        bispec (2D np.array): Bispectrum matrix (complex)
        freqs (1D np.array): Frequency axis in Hz
    """
    step = nfft - noverlap
    N = len(signal)

    # Define window
    if window_type == 'hann':
        window = np.hanning(nfft)
    elif window_type == 'rectangular':
        window = np.ones(nfft)
    else:
        raise ValueError("Unsupported window type. Use 'hann' or 'rectangular'.")

    # Initialize bispectrum matrix
    bispec = np.zeros((nfft, nfft), dtype=complex)
    num_segments = (N - nfft) // step+1

    for i in range(num_segments):
        start = i * step
        x_seg = signal[start:start + nfft] * window
        X = np.fft.fft(x_seg, n=nfft)
        for f1 in range(nfft):
            for f2 in range(nfft):
                f3 = (f1 + f2) % nfft
                bispec[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])

    bispec /= max(1, num_segments)
    freqs = np.fft.fftfreq(nfft, d=1/fs)

    if directory is not None:
            save_dir=directory[1]
            subj_name=directory[0]
            filename = os.path.join(save_dir, f'{subj_name}_bispectrum.csv')
            
            save_bispectrum_to_csv(bispec, freqs, filename)

    return bispec, freqs

########################################################################
def compute_avg_bispectrum(data_all, event, labels, fs=256, nfft=128, window_type='hann', noverlap=64, save=False, directory=None):
    """
    Compute the average bispectrum over multiple trials.

    Parameters
    ----------
    data : array_like, shape (n_trials, n_samples)
        Time-series data for each trial.
    fs : float, optional
        Sampling frequency (Hz). Default is 256.
    nfft : int, optional
        Number of FFT points. Default is 128.
    window_type : str, optional
        Type of window to apply. Default is 'hann'.
    noverlap : int, optional
        Number of points to overlap between segments. Default is 64.

    Returns
    -------
    B_avg : ndarray, shape (nfft, nfft)
        The trial-averaged bispectrum matrix.
    freqs : ndarray, shape (nfft,)
        The frequency vector corresponding to the bispectrum.
    """

    mask = labels == event
    data = data_all[mask]
    
    n_trials = data.shape[0]
    B_sum = None
    freqs = None

    for trial in range(n_trials):
        # print(trial)
        sig = data[trial, :]
        B, freqs = compute_bispectrum(sig, fs=fs, nfft=nfft,
                                      window_type=window_type,
                                      noverlap=noverlap)
        if B_sum is None:
            B_sum = np.zeros_like(B, dtype=np.complex128)
        B_sum += B
        
        if save==True:
            save_dir=directory[1]
            subj_name=directory[0]
            filename = os.path.join(save_dir, f'{subj_name}_bispectrum_ch{directory[2]}_event{event}_trial{trial}.csv')
            
            save_bispectrum_to_csv(B, freqs, filename)

    # average bispectrum
    B_avg = B_sum / n_trials
    return B_avg, freqs

################################################################################
def compute_bicoherence(signal, fs=256, nfft=128, window_type='hann', noverlap=64):
    """
    Compute biocoherence of a real-valued signal using segment averaging.

    Parameters:
        signal (1D np.array): Input signal
        fs (int): Sampling frequency
        nfft (int): FFT size
        window_type (str): Type of window to use ('hann', 'rectangular')
        noverlap (int): Number of overlapping samples

    Returns:
        bicoherence (2D np.array): Squared bicoherence matrix (real, [0, 1])
        freqs (1D np.array): Frequency axis in Hz
    """
    step = nfft - noverlap
    N = len(signal)

    # Define window
    if window_type == 'hann':
        window = np.hanning(nfft)
    elif window_type == 'rectangular':
        window = np.ones(nfft)
    else:
        raise ValueError("Unsupported window type. Use 'hann' or 'rectangular'.")

    num_segments = (N - nfft) // step+1
    bispec = np.zeros((nfft, nfft), dtype=complex)
    denom1 = np.zeros((nfft, nfft))
    denom2 = np.zeros((nfft, nfft))

    for i in range(num_segments):
        start = i * step
        x_seg = signal[start:start + nfft] * window
        X = np.fft.fft(x_seg, n=nfft)

        for f1 in range(nfft):
            for f2 in range(nfft):
                f3 = (f1 + f2) % nfft
                product = X[f1] * X[f2]
                bispec[f1, f2] += product * np.conj(X[f3])
                denom1[f1, f2] += np.abs(product) ** 2
                denom2[f1, f2] += np.abs(X[f3]) ** 2

    bispec /= max(1, num_segments)
    denom1 /= max(1, num_segments)
    denom2 /= max(1, num_segments)

    # Compute squared bicoherence
    # bicoherence = np.abs(bispec) ** 2 / (denom1 * denom2 + 1e-12)  # add small term to avoid /0
    bicoherence = bispec /np.sqrt(denom1 * denom2 + 1e-12) # keep it complex! |bicoherence| is exactly √(my previous value); still bounded 0-1.
    freqs = np.fft.fftfreq(nfft, d=1/fs)

    # np.angle(b_complex) (or b_complex.imag) retains the phase term that flips by 180 ° when you swap channel order.

    return bicoherence, freqs
##################################################################################################
def extract_band_bicoherence(data, ch_names, ch1, ch2, fs, freq_bands, nfft=128, window_type='hann', noverlap=64):
    """
    Compute band-wise (cross-)bicoherence features between two channels.
    
    Parameters:
    - ch1, ch2: arrays of time-domain signals (same length)
    - fs: sampling frequency
    - freq_bands: dict of band name -> (low_freq, high_freq)
    - nfft, window_type, noverlap: parameters passed to the bicoherence functions
    
    Returns:
    - features: dict keyed by "band1_band2", each a dict of summary metrics
    """
    # Choose bispectrum vs cross-bispectrum
    if np.array_equal(ch1, ch2):
        signal = data[:, ch_names.index(ch1)]
        bic, freqs = compute_bicoherence(signal, fs=fs, nfft=nfft, window_type=window_type, noverlap=noverlap)
    else:
        x = data[:, ch_names.index(ch1)]
        y = data[:, ch_names.index(ch2)]
        bic, freqs = compute_cross_bicoherence(x, y, fs=fs, nfft=nfft, window_type=window_type, noverlap=noverlap)
    
    features = {}
    eps = 1e-12
    
    # Precompute band indices
    band_indices = {
        name: np.where((freqs >= low) & (freqs <= high))[0]
        for name, (low, high) in freq_bands.items()
    }
    
    for band1, idx1 in band_indices.items():
        for band2, idx2 in band_indices.items():
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            
            # Extract submatrix for this band pair
            submat = bic[np.ix_(idx1, idx2)]
            flat = submat.flatten()
            
            # Basic features
            mean_val = np.mean(flat)
            max_val = np.max(flat)
            sum_val = np.sum(flat)
            
            # Peak frequency location
            max_idx = np.unravel_index(np.argmax(submat), submat.shape)
            peak_f1 = freqs[idx1][max_idx[0]]
            peak_f2 = freqs[idx2][max_idx[1]]
            
            # Entropy of the normalized bicoherence distribution
            p = flat / (flat.sum() + eps)
            entropy = -np.sum(p * np.log(p + eps))
            
            key = f"{band1}_{band2}"
            features[key] = {
                "mean_bicoherence": mean_val,
                "max_bicoherence": max_val,
                "sum_bicoherence": sum_val,
                "peak_freqs": (peak_f1, peak_f2),
                "entropy": entropy
            }
    
    return features


