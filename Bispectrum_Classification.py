def run_multiclass_classification_with_folds(X_class0, X_class1, X_class2, folds, estimator=None):
    """
    Runs multiclass classification using predefined folds.

    Parameters:
    - X_class0, X_class1, X_class2: np.ndarray, shape (n_trials, features) for each class
    - folds: list of dicts, each containing 'train' and 'test' indices (same length for all classes)
    - estimator: sklearn classifier (default = RandomForestClassifier)

    Returns:
    - DataFrame with accuracy, auc, recall across folds
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=140, random_state=42)

    accuracies, aucs, recalls = [], [], []

    for i in range(len(folds)):
        fold= folds[i]
        tr_idx, te_idx = fold['train'], fold['test']

        # --- Build train/test sets
        X_train = np.vstack([X_class0[tr_idx], X_class1[tr_idx], X_class2[tr_idx]])
        y_train = np.concatenate([
            np.zeros(len(tr_idx), dtype=int),
            np.ones(len(tr_idx), dtype=int),
            np.full(len(tr_idx), 2, dtype=int)
        ])

        X_test = np.vstack([X_class0[te_idx], X_class1[te_idx], X_class2[te_idx]])
        y_test = np.concatenate([
            np.zeros(len(te_idx), dtype=int),
            np.ones(len(te_idx), dtype=int),
            np.full(len(te_idx), 2, dtype=int)
        ])

        # --- Train fresh classifier
        clf = clone(estimator)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # --- Metrics
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average="macro")
        # AUC needs one-hot labels
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")

        accuracies.append(acc)
        recalls.append(rec)
        aucs.append(auc)

        print(f"Fold {i}: Acc={acc:.4f}, Recall={rec:.4f}, AUC={auc:.4f}")

    # --- Summary
    def summarize(values, name):
        return (f"{name}: Avg = {np.mean(values):.4f} | "
                f"Std = {np.std(values):.4f} | "
                f"Max = {np.max(values):.4f}")

    print(summarize(accuracies, "Accuracy"))
    print(summarize(aucs, "AUC"))
    print(summarize(recalls, "Recall"))

    return pd.DataFrame({
        "accuracies": accuracies,
        "aucs": aucs,
        "recalls": recalls
    })

#############################################################
def run_binary_classification_with_folds(X_class0, X_class1, folds, estimator=None):
    """
    Runs binary classification using predefined folds.
    
    Parameters:
    - X_class0: np.ndarray, shape (n_trials, features) for class 0
    - X_class1: np.ndarray, shape (n_trials, features) for class 1
    - folds: list of dicts, each containing 'train' and 'test' indices
    - estimator: sklearn classifier (default = RandomForestClassifier)
    
    Returns:
    - metrics dict with lists of results across folds
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    accuracies, aucs, recalls = [], [], []
    
    for i in range(len(folds)):
        fold= folds[i]
        tr_idx, te_idx = fold['train'], fold['test']
        
         # Separate train/test from each class
        X_train = np.vstack([X_class0[tr_idx], X_class1[tr_idx]])
        y_train = np.array([0] * len(tr_idx) + [1] * len(tr_idx))

        X_test = np.vstack([X_class0[te_idx], X_class1[te_idx]])
        y_test = np.array([0] * len(te_idx) + [1] * len(te_idx))

        # Train fresh classifier
        clf = clone(estimator)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Safe predict_proba
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)
            if 1 in clf.classes_:
                pos_index = list(clf.classes_).index(1)
                y_prob = proba[:, pos_index]
            else:
                y_prob = np.zeros(len(X_test))
        else:
            y_prob = None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None and len(np.unique(y_test)) > 1 else float("nan")

        accuracies.append(acc)
        recalls.append(rec)
        aucs.append(auc)

        print(f"Fold {i}: Acc={acc:.4f}, Recall={rec:.4f}, AUC={auc:.4f}")

    # Aggregate summary
    def summarize(values, name):
        return (f"{name}: Avg = {np.mean(values):.4f} | "
                f"Std = {np.std(values):.4f} | "
                f"Max = {np.max(values):.4f}")

    print(summarize(accuracies, "Accuracy"))
    print(summarize(aucs, "AUC"))
    print(summarize(recalls, "Recall"))

    return pd.DataFrame({
        "accuracies": accuracies,
        "aucs": aucs,
        "recalls": recalls
    })

########################################################################################################
def compute_permutation_importance(
    feature_matrix1, feature_matrix2,
    ch_names, band_pairs, feature_names,
    n_repeats=10, random_state=0, test_size=0.2,
    scoring='accuracy',
    return_labels=True
):
    """
    Train a Random Forest on two conditions and compute permutation feature importance.

    Parameters:
    - feature_matrix1, feature_matrix2: ndarrays of shape (n_samples, features)
    - ch_names: list of channel names
    - band_pairs: list of string band-pairs like 'alpha–beta'
    - feature_names: list of feature names (e.g., 'mean_bicoh', 'entropy_phase', etc.)
    - n_repeats: number of times to permute each feature
    - scoring: metric for performance drop
    - return_labels: if True, return human-readable labels

    Returns:
    - importances_mean_sorted
    - importances_std_sorted
    - baseline_accuracy
    - feature_indices_sorted
    - feature_labels_sorted (if return_labels=True)
    """
    # Stack and label
    X = np.vstack([feature_matrix1, feature_matrix2])
    y = np.array([0] * len(feature_matrix1) + [1] * len(feature_matrix2))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # Baseline accuracy
    y_pred = clf.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)

    # Permutation importance
    perm_imp = permutation_importance(
        clf, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring
    )

    importances_mean = perm_imp.importances_mean
    importances_std = perm_imp.importances_std
    feature_indices = np.arange(len(importances_mean))

    # Sort
    sorted_idx = np.argsort(importances_mean)[::-1]
    importances_mean_sorted = importances_mean[sorted_idx]
    importances_std_sorted = importances_std[sorted_idx]
    feature_indices_sorted = feature_indices[sorted_idx]

    if return_labels:
        labels_sorted = [get_feature_label(i, ch_names, band_pairs, feature_names) for i in feature_indices_sorted]
        return importances_mean_sorted, importances_std_sorted, baseline_accuracy, feature_indices_sorted, labels_sorted

    return importances_mean_sorted, importances_std_sorted, baseline_accuracy, feature_indices_sorted
  #####################################################################################################
  ######################################################################################################
def get_feature_index(channel, band_pair, feature_name, ch_names, band_pairs, feature_names):
    """
    Reverse lookup: from (channel, band_pair, feature) to flat index.
    """
    C = len(ch_names)
    B = len(band_pairs)
    F = len(feature_names)

    ch_idx = ch_names.index(channel)
    band_idx = band_pairs.index(band_pair)
    feat_idx = feature_names.index(feature_name)

    return ch_idx * (B * F) + band_idx * F + feat_idx

def recover_feature_indices_from_csv(csv_path, ch_names, band_pairs, feature_names, importance_threshold=0.0):
    """
    Read CSV and return recovered indices of features with importance > threshold.
    """
    df = pd.read_csv(csv_path)

    # Optional: filter by importance threshold
    df = df[df["Importance"] > importance_threshold]
    scores = df["Importance"].values
    stds=df["std"].values

    indices = []
    labels = []

    for _, row in df.iterrows():
        ch = row["Channel"]
        bp = row["Band_Pair"]
        feat = row["Feature"]

        idx = get_feature_index(ch, bp, feat, ch_names, band_pairs, feature_names)
        indices.append(idx)
        labels.append((ch, bp, feat))

    return indices, labels, scores, stds


############################################################################################################
def save_importance_to_csv(importances_mean_sorted, labels_sorted, std_sorted, filename="feature_importance.csv"):
    """
    Save permutation importance results to CSV with readable labels.

    Parameters:
    - importances_mean_sorted: ndarray of shape (n_features,), sorted importance scores
    - labels_sorted: list of tuples (channel, band_pair, feature), same order as importances
    - filename: str, output CSV file path

    Returns:
    - pandas DataFrame that was saved
    """
    # Unpack the labels
    channels, band_pairs, feature_names = zip(*labels_sorted)

    # Create DataFrame
    df = pd.DataFrame({
        "Channel": channels,
        "Band_Pair": band_pairs,
        "Feature": feature_names,
        "Importance": importances_mean_sorted,
        "std": std_sorted
    })

    # Save
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Saved feature importance to: {filename}")
    return df


def get_feature_label(index, ch_names, band_pairs, feature_names):
    """
    Decode a flattened feature index into (channel, band_pair, feature) label.
    """
    C, B, F = len(ch_names), len(band_pairs), len(feature_names)
    ch_idx = index // (B * F)
    band_idx = (index % (B * F)) // F
    feat_idx = index % F
    return (ch_names[ch_idx], band_pairs[band_idx], feature_names[feat_idx])

def compute_permutation_importance(
    feature_matrix1, feature_matrix2,
    ch_names, band_pairs, feature_names,
    n_repeats=10, random_state=0, test_size=0.2,
    scoring='accuracy',
    return_labels=True
):
    """
    Train a Random Forest on two conditions and compute permutation feature importance.

    Parameters:
    - feature_matrix1, feature_matrix2: ndarrays of shape (n_samples, features)
    - ch_names: list of channel names
    - band_pairs: list of string band-pairs like 'alpha–beta'
    - feature_names: list of feature names (e.g., 'mean_bicoh', 'entropy_phase', etc.)
    - n_repeats: number of times to permute each feature
    - scoring: metric for performance drop
    - return_labels: if True, return human-readable labels

    Returns:
    - importances_mean_sorted
    - importances_std_sorted
    - baseline_accuracy
    - feature_indices_sorted
    - feature_labels_sorted (if return_labels=True)
    """
    # Stack and label
    X = np.vstack([feature_matrix1, feature_matrix2])
    y = np.array([0] * len(feature_matrix1) + [1] * len(feature_matrix2))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # Baseline accuracy
    y_pred = clf.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)

    # Permutation importance
    perm_imp = permutation_importance(
        clf, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring
    )

    importances_mean = perm_imp.importances_mean
    importances_std = perm_imp.importances_std
    feature_indices = np.arange(len(importances_mean))

    # Sort
    sorted_idx = np.argsort(importances_mean)[::-1]
    importances_mean_sorted = importances_mean[sorted_idx]
    importances_std_sorted = importances_std[sorted_idx]
    feature_indices_sorted = feature_indices[sorted_idx]

    if return_labels:
        labels_sorted = [get_feature_label(i, ch_names, band_pairs, feature_names) for i in feature_indices_sorted]
        return importances_mean_sorted, importances_std_sorted, baseline_accuracy, feature_indices_sorted, labels_sorted

    return importances_mean_sorted, importances_std_sorted, baseline_accuracy, feature_indices_sorted



###################################################################################################
