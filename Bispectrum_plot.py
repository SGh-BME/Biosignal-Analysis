import numpy as np
import matplotlib.pyplot as plt
import importlib
import Bispectrum_computation as bispect
importlib.reload(bispect)
import Bispectrum_ClassificationV3 as biclass3
importlib.reload(biclass3)
import EEG_preprocessing as func
importlib.reload(func)
import matplotlib.ticker as mticker
import os
from collections import Counter, defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from collections import Counter
import Bicode_data
from Bicode_data import folds, ch_names, band_pairs, feature_names_new, subj_folders, PHASES, CONDITIONS, feature_names, data_path, ICA_comp_remove

#Feature Importance ###########################################################################################
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bandpair_heatmaps(top_feature_dir,
                           threshold=0.00,
                           save_path="bandpair_heatmaps.png"):
    """
    Create 2x3 grid of 5x5 heatmaps (band-pair contributions)
    for each condition (pen vs NM, bottle vs NM, bottle vs pen)
    and phase (plan, move).
    
    Parameters
    ----------
    top_feature_dir : str
        Directory containing *_top_features.csv files.
    threshold : float, default=0.00
        Minimum importance score to include (filter features).
    save_path : str
        Where to save the resulting figure.
    """
    plt.style.use("seaborn-v0_8-paper")
    font = {'family': 'serif', 'size': 12}
    plt.rc('font', **font)
    fontsize=16
    
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    cond_labels = {
        "pen": "Pen vs NM",
        "bottle": "Bottle vs NM",
        "bottlepen": "Bottle vs Pen"
    }
    conditions = list(cond_labels.keys())   # pen, bottle, bottlepen
    phases = ["plan", "move"]              # rows

    # ---- Collect all records ----
    rows = []
    for filename in os.listdir(top_feature_dir):
        if not filename.endswith("_top_features.csv"):
            continue
        try:
            parts = filename.replace(".csv", "").split("_")
            subj_id = int(parts[1])
            cond = parts[2].lower()
            phase = parts[3].lower()
        except Exception as e:
            print(f"⚠️ Skipping {filename} — parse error: {e}")
            continue

        csv_path = os.path.join(top_feature_dir, filename)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Failed to read {csv_path}: {e}")
            continue

        # normalize band-pair strings
        bp_norm = (df["Band_Pair"].astype(str)
                   .str.lower()
                   .str.replace("–", "-", regex=False)
                   .str.replace("—", "-", regex=False)
                   .str.replace("_", "-", regex=False)
                   .str.strip())

        df = df.assign(Band_Pair=bp_norm,
                       Subject=subj_id,
                       Condition=cond,
                       Phase=phase)

        # apply threshold filter
        df = df[df["Importance"] > threshold]

        rows.append(df[["Subject", "Condition", "Phase", "Band_Pair", "Importance"]])

    if not rows:
        raise ValueError("No valid records found!")

    df_all = pd.concat(rows, ignore_index=True)

    # ---- Aggregate: sum per subject × cond × phase × band-pair ----
    subj_pair_sum = (
        df_all.groupby(["Condition", "Phase", "Subject", "Band_Pair"], as_index=False)["Importance"]
        .sum()
        .rename(columns={"Importance": "pair_sum"})
    )

    # ---- Average across subjects ----
    group_mean = (
        subj_pair_sum.groupby(["Condition", "Phase", "Band_Pair"], as_index=False)["pair_sum"]
        .mean()
    )

    # ---- Normalize to percentages ----
    group_mean["percent"] = (
        group_mean.groupby(["Condition", "Phase"])["pair_sum"]
        .transform(lambda s: 100 * s / s.sum())
    )

    # ---- Utility to build 5x5 matrix ----
    def make_matrix(cond, phase):
        sub = group_mean[(group_mean["Condition"] == cond) &
                         (group_mean["Phase"] == phase)]
        mat = np.zeros((5,5))
        for i, d in enumerate(bands):
            for j, r in enumerate(bands):
                val = sub.loc[sub["Band_Pair"] == f"{d}-{r}", "percent"]
                mat[i, j] = float(val.iloc[0]) if not val.empty else 0.0
        return mat

    # ---- Build matrices ----
    mats = {(ph, co): make_matrix(co, ph) for ph in phases for co in conditions}
    vmax = max(mat.max() for mat in mats.values())

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    plt.style.use("seaborn-v0_8-paper")

    for i, ph in enumerate(phases):
        for j, co in enumerate(conditions):
            ax = axes[i, j]
            mat = mats[(ph, co)]
            hm = sns.heatmap(mat, ax=ax, cmap="RdBu_r", vmin=0, vmax=vmax,
                        annot=True, fmt=".1f",
                        xticklabels=bands, yticklabels=bands,
                        cbar=False,
                        annot_kws={"size": fontsize})# colorbar only on rightmost
            # if j == 2:  # rightmost column
            #     cbar = ax.collections[0].colorbar
            #     cbar.ax.tick_params(labelsize=fontsize)  # change tick font
            #     cbar.set_label("Importance (%)", fontsize=fontsize)

            ax.tick_params(axis='y', labelsize=fontsize+4)
            ax.tick_params(axis='x', labelsize=fontsize+4)
            # ax.set_xticklabels(bands, fontsize=fontsize+4, rotation=0, ha="right")
            # ax.set_yticklabels(bands, fontsize=fontsize+4, rotation=0)

            # ax.set_title(f"{cond_labels[co]} – {ph.capitalize()}",
            #              fontsize=13, fontweight="bold")
    axes[1,0].set_xlabel("Follower band", fontsize=fontsize+6)
    axes[1,1].set_xlabel("Follower band", fontsize=fontsize+6)
    axes[1,2].set_xlabel("Follower band", fontsize=fontsize+6)
    axes[0,0].set_ylabel("Driver band", fontsize=fontsize+6)
    axes[1,0].set_ylabel("Driver band", fontsize=fontsize+6)
    
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
    cbar = fig.colorbar(hm.collections[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label("Importance (%)", fontsize=fontsize)

    # Panel labels for the first row only
    panel_labels = ['a)', 'b)', 'c)']
    for ax, label in zip(axes[0], panel_labels):   # axes[0] = first row
        ax.text(0.02, 1.01, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="bottom", ha="left")



    # panel_labels = ['a)', 'b)', 'c)']
    # for ax, label in zip(axes, panel_labels):
    #     ax.text(0.028, 0.97, label, transform=ax.transAxes,
    #             fontsize=fontsize, fontweight="bold", va="top", ha="right")

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return group_mean



#--------------------------------------------------------------------------------------------------------
def plot_bandpair_count_distributions(top_bandpairs, subject_summary_df):
    # Build the DataFrame with band pair counts from subject_summary_df
    records = []
    
    for key, df in subject_summary_df.items():
        if "Band_Pair" not in df.columns:
            continue  # skip if column missing
    
        for band_pair in top_bandpairs:
            count = df["Band_Pair"].get(band_pair, 0)
            if count > 0:
                records.append([key, band_pair, count])
    
    # Create dataframe
    bandpair_count_df = pd.DataFrame(records, columns=["Subject_Phase_Cond", "Band_Pair", "Count"])
    bandpair_count_df[["Subject", "Phase", "Condition"]] = bandpair_count_df["Subject_Phase_Cond"].str.extract(
        r"subject_(\d+)_(plan|move)_(.+)"
    )
    df = bandpair_count_df

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for ax, phase in zip(axes, ["plan", "move"]):
        subset = df[df["Phase"] == phase]

        # Violin plot with hue = Band_Pair (needed for palette)
        sns.violinplot(
            data=subset,
            x="Band_Pair", y="Count",
            ax=ax, inner="box", cut=0,
            linewidth=2, width=0.8,
            density_norm="width",
            hue="Band_Pair", palette="Set2",
            legend=False  # no legend needed
        )

        # Overlay stripplot (same hue)
        sns.stripplot(
            data=subset,
            x="Band_Pair", y="Count",
            ax=ax, hue="Band_Pair", palette="Set2",
            size=3, jitter=True, dodge=False, alpha=0.6,
            legend=False
        )

        ax.set_title(f"{phase.capitalize()} Phase", fontsize=14)
        ax.set_xlabel("Band-Pair", fontsize=12)
        ax.set_ylabel("Feature Count", fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
#---------------------------------------------------------------------------------------------------
# Plot: Violin + Strip of Importance by Band-Pair (Plan vs Move)
def plot_bandpair_score_distributions(top_feature_dir, ch_names=ch_names, band_pairs=band_pairs, feature_names_new=feature_names_new, importance_threshold=0.00):
        # Collect all top features across files
    all_records = []
    for filename in os.listdir(top_feature_dir):
        if not filename.endswith("_top_features.csv"):
            continue
    
        # ---- Parse metadata from filename ----
        try:
            parts = filename.replace(".csv", "").split("_")
            subj_id = int(parts[1])
            cond_key = parts[2].lower()  # e.g., bottlepen / pen / bottle
            phase = parts[3].lower()     # plan or move
        except Exception as e:
            print(f"Skipping {filename} — failed to parse: {e}")
            continue
    
        subj_name = f"subject_{subj_id}"
        csv_path = os.path.join(top_feature_dir, filename)
    
        # ---- Load top features from CSV ----
        try:
            top_idx, top_labels, top_scores, top_stds = biclass3.recover_feature_indices_from_csv(
                csv_path=csv_path,
                ch_names=ch_names,
                band_pairs=band_pairs,
                feature_names=feature_names_new,
                importance_threshold=importance_threshold
            )
        except Exception as e:
            print(f"❌ Failed to read {csv_path}: {e}")
            continue
    
        # ---- Collect records ----
        for (ch, bp, feat), score, std in zip(top_labels, top_scores, top_stds):
            all_records.append({
                "Subject": subj_name,
                "Condition": cond_key,
                "Phase": phase,
                "Channel": ch,
                "Band_Pair": bp,
                "Feature": feat,
                "Importance": score*10,
                "Std": std*10
            })
    
    # Build DataFrame of all top features
    top_features_df = pd.DataFrame(all_records)
    
    df = top_features_df
    df["Band_Pair"] = df["Band_Pair"].astype(
    CategoricalDtype(categories=band_pairs, ordered=True))

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    fontsize=16
    
    plt.style.use("seaborn-v0_8-paper")
    font = {'family': 'serif', 'size': 12}
    plt.rc('font', **font)

    for ax, phase in zip(axes, ["plan", "move"]):
        subset = df[df["Phase"] == phase]

        sns.violinplot(
            data=subset,
            x="Band_Pair", y="Importance",
            ax=ax, inner="box", cut=0,
            linewidth=1.3, width=0.9,
            density_norm="width",
            hue="Band_Pair", palette="Set2", order=band_pairs
        )

        sns.stripplot(
            data=subset,
            x="Band_Pair", y="Importance",
            ax=ax, color="black", size=3,
            jitter=True, dodge=False, alpha=0.5
        )

        # ax.set_title(f"{phase.capitalize()} Phase", fontsize=14)
        ax.set_xlabel("Band-Pair", fontsize=fontsize+6)
        ax.set_ylabel("Importance Score", fontsize=fontsize+6)
        ax.tick_params(axis='x', rotation=90, labelsize=fontsize+4)
        ax.tick_params(axis='y', labelsize=fontsize+4)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-0.9, len(band_pairs)-0.5)
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('center')  # Already default, but safe
            label.set_x(label.get_position()[0] - 0.9)  # Shift label slightly left

        
    panel_labels = ['a)', 'b)']
    for ax, label in zip(axes, panel_labels):
        ax.text(0.028, 0.97, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="top", ha="right")

    plt.tight_layout()
    fig.savefig(f'Bispectrum_september/bandpair_score_distributions_globalUsage.png', dpi=300, bbox_inches="tight")
    plt.show()

# --------------------------------------------------------------------------------------------------------
def plot_bandpair_score_distributions_grid(top_feature_dir,
                                           ch_names=ch_names,
                                           band_pairs=band_pairs,
                                           feature_names_new=feature_names_new,
                                           importance_threshold=0.00):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    import os
    
    # ---- Collect all features ----
    all_records = []
    for filename in os.listdir(top_feature_dir):
        if not filename.endswith("_top_features.csv"):
            continue

        try:
            parts = filename.replace(".csv", "").split("_")
            subj_id = int(parts[1])
            cond_key = parts[2].lower()     # bottle / pen / bottlepen
            phase = parts[3].lower()        # plan / move
        except Exception as e:
            print(f"Skipping {filename} — parse error: {e}")
            continue

        subj_name = f"subject_{subj_id}"
        csv_path = os.path.join(top_feature_dir, filename)

        try:
            top_idx, top_labels, top_scores, top_stds = biclass3.recover_feature_indices_from_csv(
                csv_path=csv_path,
                ch_names=ch_names,
                band_pairs=band_pairs,
                feature_names=feature_names_new,
                importance_threshold=importance_threshold
            )
        except Exception as e:
            print(f"❌ Failed to read {csv_path}: {e}")
            continue

        for (ch, bp, feat), score, std in zip(top_labels, top_scores, top_stds):
            all_records.append({
                "Subject": subj_name,
                "Condition": cond_key,
                "Phase": phase,
                "Channel": ch,
                "Band_Pair": bp,
                "Feature": feat,
                "Importance": score*10,
                "Std": std*10
            })

    df = pd.DataFrame(all_records)
    df["Band_Pair"] = df["Band_Pair"].astype(
        CategoricalDtype(categories=band_pairs, ordered=True))

    # ---- Set up plotting ----
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharey=True, sharex=True)  # 3 conditions × 2 phases
    fontsize = 16
    plt.style.use("seaborn-v0_8-paper")
    font = {'family': 'serif', 'size': 12}
    plt.rc('font', **font)

    conditions = ["bottle", "pen", "bottlepen"]
    phases = ["plan", "move"]

    for i, cond in enumerate(conditions):
        for j, phase in enumerate(phases):
            ax = axes[i, j]
            subset = df[(df["Condition"] == cond) & (df["Phase"] == phase)]

            if subset.empty:
                ax.set_visible(False)
                continue

            sns.violinplot(
                data=subset,
                x="Band_Pair", y="Importance",
                ax=ax, inner="box", cut=0,
                linewidth=1.3, width=0.9,
                density_norm="width",
                hue="Band_Pair", palette="Set2", order=band_pairs,
                legend=False
            )

            sns.stripplot(
                data=subset,
                x="Band_Pair", y="Importance",
                ax=ax, color="black", size=3,
                jitter=True, dodge=False, alpha=0.5
            )

            # ax.set_title(f"{cond.capitalize()} – {phase.capitalize()}",
            #              fontsize=fontsize+2, fontweight="bold")
            ax.set_xlabel("Band-Pair", fontsize=fontsize+6)
            ax.set_ylabel("Importance Score", fontsize=fontsize+6)
            ax.tick_params(axis='x', rotation=90, labelsize=fontsize+4)
            ax.tick_params(axis='y', labelsize=fontsize+4)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(-0.9, len(band_pairs)-0.5)
            
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    axes = axes.flatten()  # <-- FIX
    for ax, label in zip(axes, panel_labels):
        ax.text(
        0.02, 0.97, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight='bold',
        va='top',
        ha='left')


    plt.tight_layout()
    fig.savefig(f'Bispectrum_september/bandpair_score_distributions_2x3.png',
                dpi=300, bbox_inches="tight")
    plt.show()

    return fig, axes

#-------------------------------------------------------------------------------------------------------
def plot_fixed_bandpair_trends_by_phase(subject_summary_df, top_bandpairs=None, figsize=(16, 6), fontsize=14):
    """
    Plots fixed band-pair trends across subjects, split by planning and movement phases.

    Parameters:
    - subject_summary_df: dict of DataFrames (per subject-phase-condition)
    - top_bandpairs: list of fixed band-pairs to include
    - figsize: tuple for figure size
    - fontsize: base font size for labels and ticks
    """
    if top_bandpairs is None:
        top_bandpairs = [
            "gamma–beta", "beta–gamma", "beta–beta", "gamma–gamma", "theta–theta",
            "delta–delta", "delta–beta", "beta–delta", "beta–alpha", "delta–theta"
        ]

    # Step 1: Rebuild records and construct dataframe
    records = []
    for key, df in subject_summary_df.items():
        for band_pair in df["Band_Pair"].dropna().index:
            count = df["Band_Pair"].get(band_pair, 0)
            if count > 0:
                records.append([key, band_pair, count])

    bandpair_df = pd.DataFrame(records, columns=["Subject_Phase_Cond", "Band_Pair", "Count"])
    bandpair_df[["Subject", "Phase", "Condition"]] = bandpair_df["Subject_Phase_Cond"].str.extract(
        r"subject_(\d+)_(plan|move)_(.+)")

    # Filter to only include fixed band-pairs
    plot_df = bandpair_df[bandpair_df["Band_Pair"].isin(top_bandpairs)]

    # Step 2: Create subplots for plan and move
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, phase in zip(axes, ["plan", "move"]):
        phase_df = plot_df[plot_df["Phase"] == phase]
        sns.lineplot(
            data=phase_df,
            x="Subject", y="Count", hue="Band_Pair",
            markers=True, dashes=False, errorbar=None,
            ax=ax
        )
        ax.set_title(f"{phase.capitalize()} Phase", fontsize=fontsize+2)
        ax.set_xlabel("Subject", fontsize=fontsize)
        ax.set_ylabel("Count in Top Features" if phase == "plan" else "", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.legend(fontsize=fontsize-2, title="Band-Pair", title_fontsize=fontsize-1)

    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------
def plot_bandpair_boxplots_by_phase(subject_summary_df, top_bandpairs=None, figsize=(16, 6), fontsize=14):
    """
    Plots boxplots of band-pair feature counts across subjects, split by planning and movement phases.

    Parameters:
    - subject_summary_df: dict of DataFrames (per subject-phase-condition)
    - top_bandpairs: list of fixed band-pairs to include
    - figsize: tuple for figure size
    - fontsize: base font size for labels and ticks
    """
    if top_bandpairs is None:
        top_bandpairs = [
            "gamma–beta", "beta–gamma", "beta–beta", "gamma–gamma", "theta–theta",
            "delta–delta", "delta–beta", "beta–delta", "beta–alpha", "delta–theta"
        ]

    # Step 1: Build dataframe
    records = []
    for key, df in subject_summary_df.items():
        for band_pair in df["Band_Pair"].dropna().index:
            count = df["Band_Pair"].get(band_pair, 0)
            if count > 0:
                records.append([key, band_pair, count])

    bandpair_df = pd.DataFrame(records, columns=["Subject_Phase_Cond", "Band_Pair", "Count"])
    bandpair_df[["Subject", "Phase", "Condition"]] = bandpair_df["Subject_Phase_Cond"].str.extract(
        r"subject_(\d+)_(plan|move)_(.+)")

    # Filter to only include fixed band-pairs
    plot_df = bandpair_df[bandpair_df["Band_Pair"].isin(top_bandpairs)]

    # Step 2: Plot setup
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, phase in zip(axes, ["plan", "move"]):
        phase_df = plot_df[plot_df["Phase"] == phase]

        sns.boxplot(
            data=phase_df,
            y="Band_Pair", x="Count", hue="Band_Pair",  # <--- Fix here
            ax=ax, orient="h",
            palette="Blues", linewidth=1.2, fliersize=3,
            legend=False  # <--- Suppress duplicate legend
        )

        ax.set_title(f"{phase.capitalize()} Phase", fontsize=fontsize+2)
        ax.set_xlabel("Count in Top Features", fontsize=fontsize)
        if phase == "plan":
            ax.set_ylabel("Band Pair", fontsize=fontsize)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    plt.show()
#----------------------------------------------------------------------------------------
def plot_all_top_n_barplots(global_summary_df, top_n=10, figsize=(18, 5)):
    """
    Plots top-N bar plots for Channel, Band_Pair, and Feature in a 1-row, 3-column layout,
    with manuscript-quality formatting.
    """
    plt.style.use("seaborn-v0_8-paper")
    font = {'family': 'serif', 'size': 12}
    plt.rc('font', **font)
    fontsize = 16

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    categories = ["Channel", "Band_Pair", "Feature"]
    titles = ["Top Channels", "Top Band-Pairs", "Top Feature Types"]

    for ax, category, title in zip(axes, categories, titles):
        top_items = global_summary_df[category].head(top_n)

        bars = ax.barh(top_items.index[::-1], top_items["Count"][::-1],
                       color="#4C72B0", edgecolor="black")

        ax.set_xlabel("Count", fontsize=fontsize+6)
        ax.tick_params(axis='y', labelsize=fontsize+4)
        ax.tick_params(axis='x', labelsize=fontsize+4)

        # Clean axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add count labels with dynamic spacing
        xlim = ax.get_xlim()[1]
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01 * xlim, bar.get_y() + bar.get_height()/2,
                    f"{int(width)}", va='center', fontsize=fontsize)

    panel_labels = ['a)', 'b)', 'c)']
    for ax, label in zip(axes, panel_labels):
        ax.text(0.07, 1.04, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="top", ha="right")

    plt.tight_layout()
    fig.savefig(f'Bispectrum_september/Top_feature_channel_bandpair_globalUsage.png', dpi=300, bbox_inches="tight")
    
    plt.show()
    
#------------------------------------------------------------------------------------
def summarize_top_feature_usage(label_dir):
    """
    Reads all *_topfeat_labels.csv files and summarizes counts of Channels, Band_Pairs, and Features.
    
    Returns:
    - subject_summary: dict → {subject_condition_phase: DataFrame of counts}
    - global_summary: dict → {category: Counter of all values across all files}
    """
    subject_summary = {}
    global_channel_counter = Counter()
    global_band_counter = Counter()
    global_feat_counter = Counter()

    for fname in os.listdir(label_dir):
        if not fname.endswith("_topfeat_labels.csv"):
            continue

        # Extract subject / phase / condition from filename
        parts = fname.replace(".csv", "").split("_")
        subject = parts[1]
        phase = parts[2]
        condpair = "_".join(parts[3:-2]) if "topfeat" in parts[-2] else parts[3]

        key = f"subject_{subject}_{phase}_{condpair}"
        path = os.path.join(label_dir, fname)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"❌ Could not read {fname}: {e}")
            continue

        ch_counts = Counter(df["Channel"])
        bp_counts = Counter(df["Band_Pair"])
        feat_counts = Counter(df["Feature"])

        # Save per-subject summary
        summary_df = pd.DataFrame({
            "Channel": pd.Series(ch_counts),
            "Band_Pair": pd.Series(bp_counts),
            "Feature": pd.Series(feat_counts)
        }).fillna(0).astype(int)

        subject_summary[key] = summary_df

        # Update global counters
        global_channel_counter.update(ch_counts)
        global_band_counter.update(bp_counts)
        global_feat_counter.update(feat_counts)

    # Global summaries as DataFrames
    global_summary = {
        "Channel": pd.DataFrame.from_dict(global_channel_counter, orient='index', columns=["Count"]).sort_values("Count", ascending=False),
        "Band_Pair": pd.DataFrame.from_dict(global_band_counter, orient='index', columns=["Count"]).sort_values("Count", ascending=False),
        "Feature": pd.DataFrame.from_dict(global_feat_counter, orient='index', columns=["Count"]).sort_values("Count", ascending=False)
    }

    return subject_summary, global_summary



##############################################################################################################
def aggregate_results_from_csv(directory, imp_test=False):
    """
    Aggregates metrics (accuracy, auc, recall) from per-subject fold CSVs.
    
    For each subject × phase × condition × band:
      - Take mean across folds
      - Take max across folds

    Then aggregates across subjects:
      - mean/std/max of subject means
      - mean/std/max of subject maxima

    So in total you get:
    1 mean (from subject means)
    2 types of max:
       max across subject means (max_accuracy)
       max across subject maxima (max_accuracy_subjmax)
       mean across subject maxima (mean_accuracy_subjmax)
    """
    records = []
    labels_info=[]
    for fname in os.listdir(directory):
        if imp_test==False:
            if fname.endswith(".csv"):
                parts = fname.replace(".csv", "").split("_")
                # Example filename: subject_2_train_test_move_bottleNM_alpha.csv
                subj_id   = parts[1]                # e.g. "2"
                phase     = parts[4]                # e.g. "move" or "plan"
                condition = parts[5]                # e.g. "bottleNM", "penNM", ...
                band      = parts[6] if len(parts) > 6 else "whole"  # e.g. "alpha"
        else:
            if fname.endswith("results.csv"):
                parts = fname.replace(".csv", "").split("_")
                subj_id   = parts[1]      # "2"
                phase     = parts[2]      # "plan"
                condition = parts[3]      # "bottlevspen"
                band = "selected"
            else:
                parts = fname.replace(".csv", "").split("_")
                subj_id   = parts[1]      # "2"
                phase     = parts[2]      # "plan"
                condition = parts[3]      # "bottlevspen"
                labels = pd.read_csv(os.path.join(directory, fname))
                # print(subj_id, phase, condition, len(labels))
                labels_info.append([subj_id, phase, condition, len(labels)])
                
                continue
                
        df = pd.read_csv(os.path.join(directory, fname))
        # print(os.path.join(directory, fname))
            
            # Per-subject mean across folds
        mean_acc = df["accuracies"].mean()
        mean_auc = df["aucs"].mean()
        mean_rec = df["recalls"].mean()
            
            # Per-subject max across folds
        max_acc = df["accuracies"].max()
        max_auc = df["aucs"].max()
        max_rec = df["recalls"].max()  
            
        records.append({
                "subject": subj_id,
                "phase": phase,
                "condition": condition,
                "band": band,
                "acc_mean": mean_acc,
                "auc_mean": mean_auc,
                "recall_mean": mean_rec,
                "acc_max": max_acc,
                "auc_max": max_auc,
                "recall_max": max_rec
            })

    df_all = pd.DataFrame(records)

    # --- Aggregate across subjects ---
    df_grouped = df_all.groupby(["phase", "condition", "band"]).agg(
        # Subject-mean based stats
        mean_accuracy=("acc_mean", "mean"),
        std_accuracy=("acc_mean", "std"),
        max_accuracy=("acc_mean", "max"),
        
        mean_auc=("auc_mean", "mean"),
        std_auc=("auc_mean", "std"),
        max_auc=("auc_mean", "max"),
        
        mean_recall=("recall_mean", "mean"),
        std_recall=("recall_mean", "std"),
        max_recall=("recall_mean", "max"),
        
        # Subject-max based stats
        mean_accuracy_subjmax=("acc_max", "mean"),
        std_accuracy_subjmax=("acc_max", "std"),
        max_accuracy_subjmax=("acc_max", "max"),
        
        mean_auc_subjmax=("auc_max", "mean"),
        std_auc_subjmax=("auc_max", "std"),
        max_auc_subjmax=("auc_max", "max"),
        
        mean_recall_subjmax=("recall_max", "mean"),
        std_recall_subjmax=("recall_max", "std"),
        max_recall_subjmax=("recall_max", "max"),
    ).reset_index()

    return df_grouped, pd.DataFrame(labels_info)




#############################################################################################################
def plot_bispectrum_from_list(bispectra_by_event, ch_idx, subj, ch_names, event_names,
                               lim1=-30, lim2=30, levels=50, cmap='RdBu_r', fontsize=16):
    """
    Plots a 1x6 layout from precomputed bispectra stored in bispectra_by_event
    Format of each entry: (B_plan, freqs_plan, B_move, freqs_move)
    """
    fig, axs = plt.subplots(1, 6, figsize=(22,4), sharex=True, sharey=True, constrained_layout=True)
    font = {'family': 'serif', 'size': 12}  # Use 'sans-serif' if required by journal
    plt.rc('font', **font)

    # Store contour sets for colorbars
    contour_sets = []

    for i, (B_plan, freqs_plan, B_move, freqs_move) in enumerate(bispectra_by_event):
        # --- Planning ---
        f1_plan = np.fft.fftshift(freqs_plan)
        X1, Y1 = np.meshgrid(f1_plan, f1_plan)
        abs_plan = np.fft.fftshift(np.abs(B_plan))

        c0 = axs[i*2].contour(X1, Y1, abs_plan, levels=levels, cmap=cmap)
        # axs[i*2].set_title(f"{event_names[i]} – Planning", fontsize=fontsize)
        contour_sets.append(c0)

        # --- Movement ---
        f1_move = np.fft.fftshift(freqs_move)
        X2, Y2 = np.meshgrid(f1_move, f1_move)
        abs_move = np.fft.fftshift(np.abs(B_move))

        c1 = axs[i*2 + 1].contour(X2, Y2, abs_move, levels=levels, cmap=cmap)
        # axs[i*2 + 1].set_title(f"{event_names[i]} – Movement", fontsize=fontsize)
        contour_sets.append(c1)

        # Common formatting
        for ax in [axs[i*2], axs[i*2 + 1]]:
            ax.set_xlabel("f1 (Hz)", fontsize=fontsize+6)
            ax.set_xlim(lim1, lim2)
            ax.set_ylim(lim1, lim2)
            ax.tick_params(axis='both', labelsize=fontsize+4)
            # ax.grid(True)

            ax.set_xticks([-20, 0, 20])
            ax.set_yticks([-20, 0, 20])
        
            # Minor ticks (hidden labels, but used for grid)
            ax.set_xticks([-10, 10], minor=True)
            ax.set_yticks([-10, 10], minor=True)
        
            # Show grid for both major + minor ticks
            ax.grid(which="both", linestyle="--", linewidth=0.6)
        
            # Hide labels on minor ticks
            ax.tick_params(which="minor", labelbottom=False, labelleft=False)

            # # Remove top and right borders
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)

    # Add ylabel only to the first subplot
    axs[0].set_ylabel("f2 (Hz)", fontsize=fontsize+6)

     # Add panel labels (a–f)
    panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    for ax, label in zip(axs, panel_labels):
        ax.text(0.1, 1.1, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="top", ha="right")

    # One colorbar for each pair (Planning + Movement)
    for i in range(3):
        pair_axes = [axs[2*i], axs[2*i+1]]
        cbar = fig.colorbar(contour_sets[2*i], ax=pair_axes, shrink=0.8, location="right", pad=0.01)
        cbar.set_label("Bispectrum Magnitude", fontsize=fontsize-1)

        # Force scientific notation (1e4 style) on colorbar ticks
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))      # always scientific notation
        formatter.set_scientific(True)         # enforce sci notation
        formatter.set_useOffset(False)
        cbar.formatter = formatter
        cbar.update_ticks()

        # Round tick labels to 2 decimals while keeping ×10^4
        def two_decimals_with_offset(x, pos):
            return f"{x/1e4:.2f}"
    
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(two_decimals_with_offset)
        )
    
        # Manually add ×10^4 text (since FuncFormatter hides it)
        cbar.ax.text(1.5, 1.02, r'$\times 10^{4}$',
                     transform=cbar.ax.transAxes,
                     fontsize=fontsize,
                     ha='left', va='bottom')

         # Set fontsize for tick labels and label
        cbar.ax.tick_params(labelsize=fontsize)

    # Super title
    # fig.suptitle(f"Bispectrum Magnitude – {ch_names[ch_idx]}, Subject: {subj}", fontsize=fontsize+4)

    # plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'AvgBispectrum_1Row3Events_{subj}_{ch_names[ch_idx]}.png', dpi=300, bbox_inches="tight")
    plt.show()





###############################################################


def plot_acc_boxplots_phases(
    df_plan, df_move, metric,sig_labels,
    bands=("delta","theta","alpha","beta","gamma"),
    tasks=("0_vs_1","0_vs_2","1_vs_2","multi"),
    titles=("a)", "b)", "c)", "d)"), #("Power G vs NM","Precision G vs NM","Power vs Precision Grasps","Multi"),
    title="Accuracy by Band across 4 tasks (Planning vs Movement)",
    figsize=(18,4.5),
    showfliers=False,
    connect_means=True,
    mean_marker="o",
    palette=None,
    point_alpha=0.6,
    jitter=0.08,
):
    fontsize=16
    """
    Paired boxplots for Planning vs Movement across bands and tasks.
    """
    # plt.style.use("seaborn-v0_8-paper")
    plt.style.use("seaborn-v0_8-white") 

    if palette is None:
        palette = {"Planning": "#56B4E9", "Movement": "#E69F00"}  # blue vs orange

    # preprocess helper
    def preprocess(df):
        d = df[df["metric"]==metric].copy()
        return (d.groupby(["subject","band","task"])["mean"]
                  .mean()
                  .reset_index())
    
    agg_plan = preprocess(df_plan)
    agg_move = preprocess(df_move)

    # fig, axes = plt.subplots(1, len(tasks), figsize=figsize, sharey=True)#, dpi=300
    # fig, axes = plt.subplots(1, 4, figsize=(16,4.5), sharey=True, dpi=300, constrained_layout=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()  # flatten to iterate easily

    font = {'family': 'serif', 'size': 12}  # Use 'sans-serif' if required by journal
    plt.rc('font', **font)

    for ax, task, t in zip(axes, tasks, titles):
        task_plan = agg_plan[agg_plan["task"]==task]
        task_move = agg_move[agg_move["task"]==task]

        data_per_band_plan, data_per_band_move = [], []
        means_plan, means_move = [], []

        for idx, b in enumerate(bands, start=1):
        
            # planning values
            vals_p = task_plan.loc[task_plan["band"]==b, "mean"].astype(float).values
            vals_p = vals_p[~np.isnan(vals_p)]
            data_per_band_plan.append(vals_p)
            means_plan.append(vals_p.mean() if vals_p.size > 0 else np.nan)

            # movement values
            vals_m = task_move.loc[task_move["band"]==b, "mean"].astype(float).values
            vals_m = vals_m[~np.isnan(vals_m)]
            data_per_band_move.append(vals_m)
            means_move.append(vals_m.mean() if vals_m.size > 0 else np.nan)

            # scatter planning
            if vals_p.size > 0:
                x = np.full(vals_p.size, idx - 0.2, dtype=float)
                if jitter:
                    rng = np.random.default_rng(42)
                    x = x + rng.uniform(-jitter, jitter, size=vals_p.size)
                ax.scatter(x, vals_p, color=palette["Planning"], alpha=point_alpha, s=15, zorder=3)

            # scatter movement
            if vals_m.size > 0:
                x = np.full(vals_m.size, idx + 0.2, dtype=float)
                if jitter:
                    rng = np.random.default_rng(43)
                    x = x + rng.uniform(-jitter, jitter, size=vals_m.size)
                ax.scatter(x, vals_m, color=palette["Movement"], alpha=point_alpha, s=15, zorder=3)
            #-------------------------------------
            label = sig_labels.get((b, task), "")
            if label:
                vals_p = data_per_band_plan[idx-1]
                vals_m = data_per_band_move[idx-1]
                if len(vals_p) > 0 and len(vals_m) > 0:
                    y_max = max(max(vals_p), max(vals_m))
                    y_pos = y_max + 0.05  # adjust vertical offset
    
                    # horizontal line
                    ax.plot([idx-0.2, idx+0.2], [y_pos, y_pos],
                            color="black", linewidth=1.2)
    
                    # label (e.g., "ns (d=0.54)" or "** (d=1.12)")
                    ax.text(idx, y_pos+0.02, label,
                            ha="center", va="bottom",
                            fontsize=10, fontweight="bold")
            #---------------------------------------

        # boxplots side by side
        positions_plan = np.arange(1, len(bands)+1) - 0.2
        positions_move = np.arange(1, len(bands)+1) + 0.2

        bp_plan = ax.boxplot(
            data_per_band_plan, positions=positions_plan,
            widths=0.35, patch_artist=True, showfliers=showfliers,
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(linewidth=1.2, color="black"),
            whiskerprops=dict(linewidth=1.2, color="black"),
            capprops=dict(linewidth=1.2, color="black")
        )
        for patch in bp_plan["boxes"]:
            patch.set_facecolor(palette["Planning"]); patch.set_alpha(0.5)

        bp_move = ax.boxplot(
            data_per_band_move, positions=positions_move,
            widths=0.35, patch_artist=True, showfliers=showfliers,
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(linewidth=1.2, color="black"),
            whiskerprops=dict(linewidth=1.2, color="black"),
            capprops=dict(linewidth=1.2, color="black")
        )
        for patch in bp_move["boxes"]:
            patch.set_facecolor(palette["Movement"]); patch.set_alpha(0.5)

        # connect means for each phase
        if connect_means:
            ax.plot(np.arange(1, len(bands)+1)-0.2, means_plan,
                    marker=mean_marker, color=palette["Planning"], linewidth=2, label="Planning" if ax==axes[0] else "")
            ax.plot(np.arange(1, len(bands)+1)+0.2, means_move,
                    marker=mean_marker, color=palette["Movement"], linewidth=2, label="Execution" if ax==axes[0] else "")

        # ax.set_title(t, fontsize=16, fontweight="bold")
        ax.set_xticks(np.arange(1, len(bands)+1))
        ax.set_xticklabels([b.capitalize() for b in bands], fontsize=fontsize+4)
        # ax.set_xlabel("Driver band", fontsize=16)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    

    axes[0].set_ylabel(f"{metric.capitalize()} (%)", fontsize=fontsize+6)
    # axes[0].legend(frameon=False, fontsize=12)

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=30)  # or 45, or 90
        ax.tick_params(axis='both', which='major', labelsize=fontsize+4)
        # ax.set_ylabel("Acc (%)", fontsize=18)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)

    # global legend (on top, centered)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=fontsize+2,  bbox_to_anchor=(0.5, 1.05))  # (x=middle, y=just above top))

    # global x-axis label
    fig.supxlabel("Driver band", fontsize=fontsize+6)

    panel_labels = ["a)", "b)", "c)", "d)"]
    for i, ax in enumerate(axes):
        ax.text(0.01, 0.95, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')


    # if title:
    #     fig.suptitle(title, y=1.04, fontsize=14, fontweight="bold")
    # plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(wspace=0.35, hspace=0.3, left=0.06, right=0.98, bottom=0.15, top=0.85)  # Manual layout
    fig.tight_layout(pad=0.8)
    fig.savefig("training_acc.png", dpi=300, bbox_inches='tight', pad_inches=0.1)# , bbox_inches='tight', pad_inches=0.6
    
    return fig, axes

#################################################################

def plot_acc_boxplots_subjectwise(
    df, metric,
    bands=("delta","theta","alpha","beta","gamma"),
    tasks=("0_vs_1","0_vs_2","1_vs_2","multi"),
    titles=["Power G vs NM","Precision G vs NM","Power vs Precision Grasps","Multi"],
    title="Accuracy by Band across 4 tasks (subject-wise)",
    figsize=(16,4.5),
    showfliers=False,
    connect_means=True,
    mean_marker="o",
    mean_color="red",
    show_points=True,
    point_color="black",
    jitter=0.08
):
    """
    Make 1x4 subplots. Each subplot = one task, each boxplot = distribution
    of subject-level accuracies for that band.
    
    Adds:
    - line connecting the band-wise means
    - scatter points for subject values
    """
    # filter to chosen metric
    d = df[df["metric"]==metric].copy()
    
    # aggregate to subject-level mean across splits
    agg = (d.groupby(["subject","band","task"])["mean"]
             .mean()
             .reset_index())

    fig, axes = plt.subplots(1,4, figsize=figsize, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    for ax, task, t in zip(axes, tasks, titles):
        task_data = agg[agg["task"]==task]
        data_per_band = []
        means = []
        for idx, b in enumerate(bands, start=1):
            vals = task_data.loc[task_data["band"]==b, "mean"].astype(float).values
            vals = vals[~np.isnan(vals)]
            data_per_band.append(vals)
            means.append(vals.mean() if vals.size > 0 else np.nan)
            
            # scatter subject dots with jitter
            if show_points and vals.size > 0:
                x = np.full(vals.size, idx, dtype=float)
                if jitter:
                    rng = np.random.default_rng(42)
                    x = x + rng.uniform(-jitter, jitter, size=vals.size)
                ax.scatter(x, vals, color=point_color, alpha=0.7, s=20, zorder=3)
        
        # boxplot
        ax.boxplot(
            data_per_band,
            labels=bands,
            showfliers=showfliers,
            patch_artist=True
        )
        
        # connect means
        if connect_means:
            x = np.arange(1, len(bands)+1)
            ax.plot(x, means, marker=mean_marker, color=mean_color, linestyle="-", linewidth=2, label="Mean")
        
        ax.set_title(t)
        ax.set_xlabel("Band")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    
    axes[0].set_ylabel(metric.capitalize())
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes
########################################################


import matplotlib.pyplot as plt
import numpy as np

def plot_acc_boxplots_subjectwise2(
    df, metric,
    bands=("delta","theta","alpha","beta","gamma"),
    tasks=("0_vs_1","0_vs_2","1_vs_2","multi"),
    titles=("Power G vs NM","Precision G vs NM","Power vs Precision Grasps","Multi"),
    title="Accuracy by Band across 4 tasks (subject-wise)",
    figsize=(16,4.5),
    showfliers=False,
    connect_means=True,
    mean_marker="o",
    mean_color="orange",
    show_points=True,
    point_color="black",
    jitter=0.08,
    palette=None
):
    """
    Publication-quality subjectwise boxplots with per-band distributions.
    """
    plt.style.use("seaborn-v0_8-paper")
    
    d = df[df["metric"]==metric].copy()
    agg = (d.groupby(["subject","band","task"])["mean"]
             .mean()
             .reset_index())

    if palette is None:
        palette = plt.cm.Set2.colors  # professional soft colors
    
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True, dpi=300)

    for ax, task, t in zip(axes, tasks, titles):
        task_data = agg[agg["task"]==task]
        data_per_band, means = [], []
        
        for idx, b in enumerate(bands, start=1):
            vals = task_data.loc[task_data["band"]==b, "mean"].astype(float).values
            vals = vals[~np.isnan(vals)]
            data_per_band.append(vals)
            means.append(vals.mean() if vals.size > 0 else np.nan)

            # scatter subject dots
            if show_points and vals.size > 0:
                x = np.full(vals.size, idx, dtype=float)
                if jitter:
                    rng = np.random.default_rng(42)
                    x = x + rng.uniform(-jitter, jitter, size=vals.size)
                ax.scatter(x, vals, color=point_color, alpha=0.6, s=15, zorder=3)

        # boxplot with colored fills
        bp = ax.boxplot(
            data_per_band,
            labels=[b.capitalize() for b in bands],
            showfliers=showfliers,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(linewidth=1.2, color="black"),
            whiskerprops=dict(linewidth=1.2, color="black"),
            capprops=dict(linewidth=1.2, color="black")
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        # connect means
        if connect_means:
            x = np.arange(1, len(bands)+1)
            ax.plot(x, means, marker=mean_marker, color=mean_color,
                    linestyle="-", linewidth=2.2, markersize=6, label="Mean")

        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.set_xlabel("Band", fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(f"{metric.capitalize()} (%)", fontsize=12)
    if connect_means:
        axes[-1].legend(loc="upper right", frameon=False, fontsize=10)

    if title:
        fig.suptitle(title, y=1.04, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes

















