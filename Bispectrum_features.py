##################################################################################################
def compute_biocoherence_features(
        data, labels, event, freq_bands,
        fs=256, nfft=128, window_type='hann', noverlap=64):
    """
    Extract magnitude- and phase-based features exclusively from the
    complex biocoherence b_cplx(f1,f2).

    Parameters
    ----------
    data        : ndarray (trials, samples, channels)
    labels      : ndarray (trials,)  trial labels
    event       : label to select (e.g. 'planning')
    freq_bands  : dict  {'theta':(4,8), 'beta':(13,25), ...}
    fs, nfft, window_type, noverlap : passed to bispect.compute_bicoherence

    Returns
    -------
    feature_matrix :  ndarray (trials, channels, band_pairs, 8)
                     [mean_bic, max_bic, sum_bic, entropy_bic,
                      mean_phase, concentration_R, circ_var, entropy_phase]
    band_pairs     :  list of (band1, band2) ordered pairs
    feature_names  :  list of 8 strings
    """

    mask       = labels == event
    data_all   = data[mask]
    n_trials, _, n_ch = data_all.shape

    bands      = list(freq_bands.keys())
    band_pairs = [(b1, b2) for b1 in bands for b2 in bands]   # ordered
    n_pairs    = len(band_pairs)

    feature_names = [
        "mean_bicoh", "max_bicoh", "sum_bicoh", "entropy_bicoh",
        "mean_phase", "concentration_R", "circ_variance", "entropy_phase"
    ]
    n_feat = len(feature_names)
    feature_matrix = np.zeros((n_trials, n_ch, n_pairs, n_feat), dtype=float)

    # --- pre-compute freq indices for every band pair ------------------
    bic_sample, freqs = bispect.compute_bicoherence(
        data_all[0, :, 0], fs=fs, nfft=nfft,
        window_type=window_type, noverlap=noverlap
    )
    band_idx = {
        (b1, b2): (
            np.where((freqs >= freq_bands[b1][0]) & (freqs <= freq_bands[b1][1]))[0],
            np.where((freqs >= freq_bands[b2][0]) & (freqs <= freq_bands[b2][1]))[0]
        )
        for b1, b2 in band_pairs
    }

    # --- main loop ------------------------------------------------------
    for t in range(n_trials):
        print(t)
        for ch in range(n_ch):
            sig          = data_all[t, :, ch]
            bic_complex, _ = bispect.compute_bicoherence(
                sig, fs=fs, nfft=nfft,
                window_type=window_type, noverlap=noverlap
            )

            for p, (b1, b2) in enumerate(band_pairs):
                idx1, idx2   = band_idx[(b1, b2)]
                sub          = bic_complex[np.ix_(idx1, idx2)].flatten()

                # magnitude features
                mag          = np.abs(sub)
                mean_bic     = mag.mean()
                max_bic      = mag.max()
                sum_bic      = mag.sum()
                p_mag        = mag / (sum_bic + 1e-12)
                entropy_bic  = -np.nansum(p_mag * np.log(p_mag + 1e-12))

                # phase features
                phase_vals   = np.angle(sub)
                mean_phase   = np.angle(np.mean(np.exp(1j * phase_vals)))
                R            = np.abs(np.mean(np.exp(1j * phase_vals)))  # concentration
                circ_var     = 1 - R
                hist, _      = np.histogram(
                                   phase_vals, bins=18, range=(-np.pi, np.pi),
                                   density=True)
                entropy_phase = -np.nansum(hist * np.log(hist + 1e-12))

                feature_matrix[t, ch, p, :] = [
                    mean_bic, max_bic, sum_bic, entropy_bic,
                    mean_phase, R, circ_var, entropy_phase
                ]

    return feature_matrix, band_pairs, feature_names
    ######################################################################################################

def preprocess_features_for_classification(feature_matrix, feature_names, flatten=False):
    """
    Preprocess a feature matrix of shape (trials, channels, band-pairs, features)
    by applying appropriate normalization techniques to each feature, and
    flattening it to shape (trials, features_flattened) for classification.

    Parameters:
    - feature_matrix: numpy array of shape (75, 16, 25, 8)
    - feature_names: list of feature names of length 8

    Returns:
    - X_classification: numpy array of shape (75, 3200)
    """
    if flatten:
        feature_matrix = feature_matrix.reshape(75,16,25,8)
        
    trials, channels, bands, num_features = feature_matrix.shape
    reshaped_matrix = np.moveaxis(feature_matrix, -1, 0)  # Shape: (8, 75, 16, 25)
    processed_features = []

    for i, name in enumerate(feature_names):
        data = reshaped_matrix[i].reshape(-1)  # Flatten (75*16*25,)
        
        if name == 'sum_bicoh':
            # Log-transform then z-score
            data = np.log1p(data)
            # scaler = StandardScaler()
            # data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        elif name == 'mean_phase':
            # Convert to sin and cos components and z-score
            sin_comp = np.sin(data)
            cos_comp = np.cos(data)
            # sin_scaled = StandardScaler().fit_transform(sin_comp.reshape(-1, 1)).flatten()
            # cos_scaled = StandardScaler().fit_transform(cos_comp.reshape(-1, 1)).flatten()
            sin_scaled = sin_comp.reshape(-1, 1).flatten()
            cos_scaled = cos_comp.reshape(-1, 1).flatten()
            sin_scaled = sin_scaled.reshape(trials, channels, bands)
            cos_scaled = cos_scaled.reshape(trials, channels, bands)
            processed_features.append(sin_scaled)
            processed_features.append(cos_scaled)
            continue  # skip appending original mean_phase
            
        
        elif name in ['concentration_R', 'circ_variance']:
            # Already in [0, 1]; z-score to center
            # scaler = StandardScaler()
            # data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            data = data.reshape(-1, 1).flatten()

        elif name in ['mean_bicoh', 'max_bicoh', 'entropy_bicoh', 'entropy_phase']:
            # Use RobustScaler to reduce outlier effects
            # scaler = RobustScaler()
            # data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            data = data.reshape(-1, 1).flatten()

        else:
            # Fallback: standard scaling
            # scaler = StandardScaler()
            # data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            data = data.reshape(-1, 1).flatten()

        # Reshape back to (75, 16, 25)
        data = data.reshape(trials, channels, bands)
        processed_features.append(data)

    # Stack processed features → shape: (75, 16, 25, num_new_features)
    final_matrix = np.stack(processed_features, axis=-1)

    # Reshape to (75, features_flattened) for classification
    X_classification = final_matrix.reshape(trials, -1)

    return X_classification
   ######################################################################################################
