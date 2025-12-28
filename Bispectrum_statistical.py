import pandas as pd
import numpy as np
import scipy.stats as stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests

def kendalls_w(matrix):
    """
    Compute Kendall's W (effect size for Friedman test).
    Input: subjects × bands matrix (numpy or pandas).
    """
    k = matrix.shape[1]  # number of bands
    n = matrix.shape[0]  # number of subjects
    ranks = stats.rankdata(matrix, axis=1)
    Rj = np.sum(ranks, axis=0)
    meanRj = np.mean(Rj)
    S = np.sum((Rj - meanRj) ** 2)
    W = 12 * S / (n**2 * (k**3 - k))
    return W

def rank_biserial(x, y):
    """
    Compute rank-biserial correlation for Wilcoxon signed-rank.
    """
    diff = x - y
    pos = np.sum(diff > 0)
    neg = np.sum(diff < 0)
    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0

def test_band_effect(df_plan, df_move, metric="acc", bands=("delta","theta","alpha","beta","gamma")):
    results = []          # Friedman results
    posthoc_results = []  # Pairwise comparisons

    for phase_name, df in zip(["planning","movement"], [df_plan, df_move]):
        df_metric = df[df["metric"] == metric]

        for task in df_metric["task"].unique():
            task_data = df_metric[df_metric["task"] == task]

            # subjects × bands matrix
            pivot = task_data.pivot_table(index="subject", columns="band", values="mean")

            # keep only available bands (some may be missing)
            available_bands = [b for b in bands if b in pivot.columns]
            pivot = pivot[available_bands].dropna(how="any")

            if pivot.shape[0] > 1 and len(available_bands) > 1:
                # Friedman test
                stat, p_val = stats.friedmanchisquare(*[pivot[b] for b in available_bands])
                W = kendalls_w(pivot.values)

                results.append({
                    "phase": phase_name,
                    "task": task,
                    "bands_tested": available_bands,
                    "friedman_stat": stat,
                    "friedman_p": p_val,
                    "kendalls_W": W
                })

                # Always run post-hoc (even if Friedman not significant)
                pairs = list(combinations(available_bands, 2))
                pvals = []
                temp_results = []
                for b1, b2 in pairs:
                    x, y = pivot[b1].values, pivot[b2].values
                    try:
                        w_stat, p = stats.wilcoxon(x, y)
                    except ValueError:
                        w_stat, p = np.nan, np.nan
                    rbc = rank_biserial(x, y)
                    temp_results.append({
                        "phase": phase_name,
                        "task": task,
                        "band1": b1,
                        "band2": b2,
                        "stat": w_stat,
                        "p_val": p,
                        "effect_size_rbc": rbc
                    })
                    pvals.append(p)

                # Correct p-values
                if len(pvals) > 0:
                    reject, pvals_corr, _, _ = multipletests(pvals, method='fdr_bh')
                    for i, _ in enumerate(temp_results):
                        temp_results[i]["p_val_FDR"] = pvals_corr[i]
                        temp_results[i]["significant"] = reject[i]

                posthoc_results.extend(temp_results)

    return pd.DataFrame(results), pd.DataFrame(posthoc_results)
#########################################################################################################
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro
from statsmodels.stats.multitest import multipletests

def paired_feature_tests_all_conditions(
    condition_data_plan: dict,
    condition_data_move: dict,
    conditions=("nm", "bottle", "pen"),
    alpha=0.05,
    fdr_method="fdr_bh",
    normality_alpha=0.05,
    reduce_trials="mean",   # "mean" (recommended) or "median"
    warn_if_zscored=True,
):
    """
    Feature-level within-subject paired tests: Planning vs Movement for each condition.

    Inputs
    ------
    condition_data_plan: dict
        e.g., {"nm": [subj0_arr, ..., subj9_arr], "bottle": [...], "pen": [...]}
        each subj_arr shape: (n_trials, 16, 25, 9)
    condition_data_move: dict
        same structure as condition_data_plan, for movement phase
    conditions: iterable
        which keys to run, default ("nm","bottle","pen")
    alpha: float
        significance threshold for FDR
    fdr_method: str
        statsmodels multipletests method, default "fdr_bh"
    normality_alpha: float
        threshold for Shapiro on paired differences
    reduce_trials: str
        "mean" or "median" across trials to get subject-level feature values
    warn_if_zscored: bool
        prints a reminder about separate z-scoring risk

    Returns
    -------
    results: dict
        results[cond] contains:
            - p_unc: (16,25,9)
            - p_fdr: (16,25,9)
            - reject: (16,25,9) boolean
            - effect: (16,25,9)
            - test_used: (16,25,9) object array
            - summary: dict with counts and min p
    """
    if warn_if_zscored:
        print("Note: If you z-scored each phase separately, mean Plan-vs-Move differences can be suppressed.")

    reducer = np.mean if reduce_trials == "mean" else np.median
    results = {}

    for cond in conditions:
        plan_list = condition_data_plan[cond]
        move_list = condition_data_move[cond]

        if len(plan_list) != len(move_list):
            raise ValueError(f"{cond}: plan subjects ({len(plan_list)}) != move subjects ({len(move_list)})")

        # Reduce trials -> subject-level matrices: (16,25,9)
        plan = np.stack([reducer(subj, axis=0) for subj in plan_list], axis=0)  # (S,16,25,9)
        move = np.stack([reducer(subj, axis=0) for subj in move_list], axis=0)  # (S,16,25,9)

        S, n_ch, n_bp, n_feat = plan.shape
        if (n_ch, n_bp, n_feat) != (16, 25, 9):
            print(f"Warning: {cond} feature shape is {(n_ch,n_bp,n_feat)} (expected (16,25,9))")

        p_unc = np.ones((n_ch, n_bp, n_feat), dtype=float)
        eff = np.zeros((n_ch, n_bp, n_feat), dtype=float)
        test_used = np.empty((n_ch, n_bp, n_feat), dtype=object)

        for ch in range(n_ch):
            for bp in range(n_bp):
                for f in range(n_feat):
                    x = plan[:, ch, bp, f]
                    y = move[:, ch, bp, f]
                    diff = y - x

                    # If all diffs are identical, tests can be degenerate
                    if np.allclose(diff, diff[0]):
                        p_unc[ch, bp, f] = 1.0
                        eff[ch, bp, f] = 0.0
                        test_used[ch, bp, f] = "degenerate"
                        continue

                    # Normality of paired differences
                    try:
                        _, p_norm = shapiro(diff)
                    except Exception:
                        p_norm = 0.0  # fall back to nonparametric if Shapiro fails

                    if p_norm > normality_alpha:
                        # Paired t-test
                        _, p = ttest_rel(y, x)
                        # Cohen's d for paired samples
                        sd = diff.std(ddof=1)
                        d = 0.0 if sd == 0 else diff.mean() / sd
                        p_unc[ch, bp, f] = float(p)
                        eff[ch, bp, f] = float(d)
                        test_used[ch, bp, f] = "t-test"
                    else:
                        # Wilcoxon signed-rank
                        try:
                            _, p = wilcoxon(y, x, zero_method="wilcox", correction=False)
                        except ValueError:
                            # e.g., all zeros after dropping ties
                            p = 1.0
                        # Simple directional effect in [-1,1] based on sign balance
                        eff_dir = (np.sum(diff > 0) - np.sum(diff < 0)) / len(diff)
                        p_unc[ch, bp, f] = float(p)
                        eff[ch, bp, f] = float(eff_dir)
                        test_used[ch, bp, f] = "wilcoxon"

        # FDR across all 3600 features for this condition
        p_flat = p_unc.reshape(-1)
        reject_flat, p_fdr_flat, _, _ = multipletests(p_flat, alpha=alpha, method=fdr_method)
        p_fdr = p_fdr_flat.reshape(n_ch, n_bp, n_feat)
        reject = reject_flat.reshape(n_ch, n_bp, n_feat)

        summary = {
            "n_subjects": int(S),
            "n_tests": int(n_ch * n_bp * n_feat),
            "min_p_unc": float(np.min(p_unc)),
            "n_sig_unc_p05": int(np.sum(p_unc < 0.05)),
            "n_sig_fdr": int(np.sum(reject)),
        }

        results[cond] = {
            "p_unc": p_unc,
            "p_fdr": p_fdr,
            "reject": reject,
            "effect": eff,
            "test_used": test_used,
            "plan_subject_mean": plan,  # (S,16,25,9)
            "move_subject_mean": move,  # (S,16,25,9)
            "summary": summary,
        }

    return results
