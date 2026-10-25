"""Statistical hypothesis testing, supervised machine learning, and time series forecasting."""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pydantic import BaseModel, Field
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_tools

RANDOM_STATE = 42


class StatTestRequest(BaseModel):
    """Specification schema for statistical hypothesis testing."""

    test_type: Literal["compare_groups", "association", "correlation", "ab_proportion"] = Field(
        ..., description="Type of statistical test"
    )
    target: str = Field(..., description="Target variable name")
    group_col: Optional[str] = Field(None, description="Grouping column for group comparison or A/B tests")
    second_col: Optional[str] = Field(None, description="Second column for correlation or association")
    alpha: float = Field(0.05, description="Significance threshold")


def _label_effect_size(name: str, val: float) -> str:
    """Assign standard rule-of-thumb qualitative benchmark to effect size value."""
    abs_val = abs(val)
    name_lower = name.lower()
    if "cohen" in name_lower or name_lower == "d":
        if abs_val < 0.2:
            return "negligible"
        elif abs_val < 0.5:
            return "small"
        elif abs_val < 0.8:
            return "medium"
        return "large"
    elif "eta" in name_lower or "epsilon" in name_lower:
        if abs_val < 0.01:
            return "negligible"
        elif abs_val < 0.06:
            return "small"
        elif abs_val < 0.14:
            return "medium"
        return "large"
    else:  # Cramer's V, Pearson r, Spearman rho, Rank-biserial
        if abs_val < 0.1:
            return "negligible"
        elif abs_val < 0.3:
            return "small"
        elif abs_val < 0.5:
            return "medium"
        return "large"


def run_stat_test(df: pd.DataFrame, req: StatTestRequest | dict) -> dict:
    """Execute hypothesis tests with assumption checking, effect sizes, CIs, and warnings."""
    if isinstance(req, dict):
        req = StatTestRequest(**req)

    # 1. Validate columns
    required_cols = [req.target]
    if req.group_col and req.group_col not in required_cols:
        required_cols.append(req.group_col)
    if req.second_col and req.second_col not in required_cols:
        required_cols.append(req.second_col)

    for c in required_cols:
        if c not in df.columns:
            return {"error": f"Column '{c}' not found in dataset."}

    # Drop NA on involved columns before running the selected test.
    sub_df = df[required_cols].dropna().copy()
    if len(sub_df) == 0:
        return {"error": f"No valid observations remaining after dropping missing values in {required_cols}."}

    warnings: list[str] = []
    alpha = req.alpha

    # -------------------------------------------------------------
    # Test Type: compare_groups (numeric target x categorical group_col)
    # -------------------------------------------------------------
    if req.test_type == "compare_groups":
        group_col = req.group_col or req.second_col
        if not group_col:
            return {"error": "Group comparison requires 'group_col'."}

        # Check unique levels
        sub_df[group_col] = sub_df[group_col].astype(str)
        unique_groups = sub_df[group_col].unique()
        k_groups = len(unique_groups)

        if k_groups > 20:
            return {"error": f"Grouping column '{group_col}' has {k_groups} distinct levels (maximum allowed is 20)."}
        if k_groups < 2:
            return {"error": f"Grouping column '{group_col}' must have at least 2 distinct groups (found {k_groups})."}

        # Validate numeric target
        if not pd.api.types.is_numeric_dtype(sub_df[req.target]):
            return {"error": f"Target column '{req.target}' must be numeric for group comparison."}

        # Check sample size per group
        group_data = {}
        for g in unique_groups:
            vals = sub_df[sub_df[group_col] == g][req.target].values
            if len(vals) < 5:
                return {"error": f"Group '{g}' has only {len(vals)} observations. Minimum required is 5."}
            group_data[g] = vals

        group_summary = [
            {
                "group": str(g),
                "n": int(len(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
            }
            for g, vals in group_data.items()
        ]

        # Check unbalanced groups
        ns = [len(v) for v in group_data.values()]
        if max(ns) / min(ns) > 3:
            warnings.append(f"Highly unbalanced group sizes (ratio {max(ns)/min(ns):.1f} > 3).")
        if min(ns) < 30:
            warnings.append(f"Small group sample size detected (minimum group n={min(ns)} < 30).")

        # -----------------------------
        # Case A: Exactly 2 groups
        # -----------------------------
        if k_groups == 2:
            g1, g2 = unique_groups[0], unique_groups[1]
            x1, x2 = group_data[g1], group_data[g2]
            n1, n2 = len(x1), len(x2)

            # Shapiro-Wilk (subsample 5000 with seed 42)
            rng = np.random.RandomState(RANDOM_STATE)
            s1_sub = rng.choice(x1, size=min(n1, 5000), replace=False)
            s2_sub = rng.choice(x2, size=min(n2, 5000), replace=False)
            _, shapiro_p1 = stats.shapiro(s1_sub)
            _, shapiro_p2 = stats.shapiro(s2_sub)
            normal = (shapiro_p1 >= 0.05) and (shapiro_p2 >= 0.05)

            # Levene test for equal variance
            _, levene_p = stats.levene(x1, x2)
            equal_var = bool(levene_p >= 0.05)

            # Mean difference and 95% CI
            mean_diff = float(np.mean(x1) - np.mean(x2))
            s1_var, s2_var = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
            se_diff = math.sqrt(s1_var / n1 + s2_var / n2)
            df_welch = (s1_var / n1 + s2_var / n2) ** 2 / (
                ((s1_var / n1) ** 2) / (n1 - 1) + ((s2_var / n2) ** 2) / (n2 - 1)
            )
            t_crit = float(stats.t.ppf(1 - alpha / 2, df_welch))
            ci_95 = [round(mean_diff - t_crit * se_diff, 4), round(mean_diff + t_crit * se_diff, 4)]

            # Pooled SD for Cohen's d
            pooled_sd = math.sqrt(((n1 - 1) * s1_var + (n2 - 1) * s2_var) / (n1 + n2 - 2)) if (n1 + n2 > 2) else 1.0
            cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

            # Decide test
            if not normal and (n1 < 30 or n2 < 30):
                test_name = "Mann-Whitney U Test"
                why_chosen = f"Non-normal distributions with sample size < 30 (Shapiro p1={shapiro_p1:.3g}, p2={shapiro_p2:.3g})."
                mwu_res = stats.mannwhitneyu(x1, x2, alternative="two-sided")
                statistic = float(mwu_res.statistic)
                p_val = float(mwu_res.pvalue)
                # Rank-biserial correlation
                rb = float(1 - (2 * statistic) / (n1 * n2))
                effect_size = {"name": "rank_biserial", "value": round(rb, 4), "label": _label_effect_size("rank_biserial", rb)}
            elif normal and equal_var:
                test_name = "Student's Two-Sample t-Test"
                why_chosen = "Both groups are normally distributed with equal variance (Levene p >= 0.05)."
                t_res = stats.ttest_ind(x1, x2, equal_var=True)
                statistic = float(t_res.statistic)
                p_val = float(t_res.pvalue)
                effect_size = {"name": "cohens_d", "value": round(cohen_d, 4), "label": _label_effect_size("cohens_d", cohen_d)}
            else:
                test_name = "Welch's t-Test"
                why_chosen = "Default robust test for two groups with unequal variances or moderate skewness."
                t_res = stats.ttest_ind(x1, x2, equal_var=False)
                statistic = float(t_res.statistic)
                p_val = float(t_res.pvalue)
                effect_size = {"name": "cohens_d", "value": round(cohen_d, 4), "label": _label_effect_size("cohens_d", cohen_d)}

            significant = bool(p_val < alpha)
            if significant and effect_size["label"] == "negligible":
                warnings.append("Statistically significant result with negligible practical effect size.")

            return {
                "test_name": test_name,
                "why_chosen": why_chosen,
                "assumptions": {"normality": normal, "equal_variance": equal_var},
                "n": int(len(sub_df)),
                "group_summary": group_summary,
                "statistic": round(statistic, 4),
                "p_value": float(p_val),
                "alpha": alpha,
                "significant": significant,
                "effect_size": effect_size,
                "ci_95": ci_95,
                "post_hoc": None,
                "warnings": warnings,
            }

        # -----------------------------
        # Case B: >= 3 groups
        # -----------------------------
        else:
            # Compute residuals
            residuals = []
            for g, vals in group_data.items():
                residuals.extend(vals - np.mean(vals))
            res_arr = np.array(residuals)

            rng = np.random.RandomState(RANDOM_STATE)
            res_sub = rng.choice(res_arr, size=min(len(res_arr), 5000), replace=False)
            _, shapiro_p = stats.shapiro(res_sub)
            normal = bool(shapiro_p >= 0.05)

            _, levene_p = stats.levene(*group_data.values())
            equal_var = bool(levene_p >= 0.05)

            # Total and Between sum of squares for eta^2
            all_vals = np.concatenate(list(group_data.values()))
            grand_mean = np.mean(all_vals)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in group_data.values())
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

            post_hoc = None

            if normal and equal_var:
                test_name = "One-Way ANOVA"
                why_chosen = "Residuals are normally distributed and group variances are homogeneous."
                f_res = stats.f_oneway(*group_data.values())
                statistic = float(f_res.statistic)
                p_val = float(f_res.pvalue)
                effect_size = {"name": "eta_squared", "value": round(eta_sq, 4), "label": _label_effect_size("eta_squared", eta_sq)}

                # Tukey HSD if significant
                if p_val < alpha:
                    warnings.append("Multi-group post-hoc analysis performed (family-wise error controlled via Tukey HSD).")
                    tukey = pairwise_tukeyhsd(endog=sub_df[req.target], groups=sub_df[group_col], alpha=alpha)
                    post_hoc = [
                        {
                            "group1": str(row[0]),
                            "group2": str(row[1]),
                            "meandiff": round(float(row[2]), 4),
                            "p_adj": round(float(row[3]), 4),
                            "lower": round(float(row[4]), 4),
                            "upper": round(float(row[5]), 4),
                            "reject": bool(row[6]),
                        }
                        for row in tukey.summary().data[1:]
                    ]
            else:
                test_name = "Kruskal-Wallis H Test"
                why_chosen = f"Normality or equal-variance assumption violated (Shapiro p={shapiro_p:.3g}, Levene p={levene_p:.3g})."
                kw_res = stats.kruskal(*group_data.values())
                statistic = float(kw_res.statistic)
                p_val = float(kw_res.pvalue)
                # Epsilon-squared
                n_total = len(all_vals)
                eps_sq = (statistic - k_groups + 1) / (n_total - k_groups) if (n_total > k_groups) else 0.0
                eps_sq = max(0.0, float(eps_sq))
                effect_size = {"name": "epsilon_squared", "value": round(eps_sq, 4), "label": _label_effect_size("epsilon_squared", eps_sq)}

                # Pairwise Mann-Whitney with Holm correction if significant
                if p_val < alpha:
                    warnings.append("Multi-group post-hoc pairwise Mann-Whitney tests performed with Holm-Bonferroni correction.")
                    p_vals_raw = []
                    pairs = []
                    group_names = list(group_data.keys())
                    for i in range(len(group_names)):
                        for j in range(i + 1, len(group_names)):
                            gA, gB = group_names[i], group_names[j]
                            mwu = stats.mannwhitneyu(group_data[gA], group_data[gB], alternative="two-sided")
                            pairs.append((gA, gB, float(np.median(group_data[gA]) - np.median(group_data[gB]))))
                            p_vals_raw.append(mwu.pvalue)

                    reject, p_adjusted, _, _ = multipletests(p_vals_raw, alpha=alpha, method="holm")
                    post_hoc = [
                        {
                            "group1": str(pairs[k][0]),
                            "group2": str(pairs[k][1]),
                            "mediandiff": round(float(pairs[k][2]), 4),
                            "p_adj": round(float(p_adjusted[k]), 4),
                            "reject": bool(reject[k]),
                        }
                        for k in range(len(pairs))
                    ]

            significant = bool(p_val < alpha)
            if significant and effect_size["label"] == "negligible":
                warnings.append("Statistically significant result with negligible practical effect size.")

            return {
                "test_name": test_name,
                "why_chosen": why_chosen,
                "assumptions": {"normality": normal, "equal_variance": equal_var},
                "n": int(len(sub_df)),
                "group_summary": group_summary,
                "statistic": round(statistic, 4),
                "p_value": float(p_val),
                "alpha": alpha,
                "significant": significant,
                "effect_size": effect_size,
                "ci_95": None,
                "post_hoc": post_hoc,
                "warnings": warnings,
            }

    # -------------------------------------------------------------
    # Test Type: association (two categoricals)
    # -------------------------------------------------------------
    elif req.test_type == "association":
        col2 = req.second_col or req.group_col
        if not col2:
            return {"error": "Categorical association test requires a second column."}

        ct = pd.crosstab(sub_df[req.target], sub_df[col2])
        r, c = ct.shape
        if r < 2 or c < 2:
            return {"error": f"Contingency table must be at least 2x2 (found {r}x{c})."}

        chi2, p_val, dof, expected = stats.chi2_contingency(ct)
        n_obs = ct.values.sum()

        # Check Fisher's exact condition: 2x2 table and any expected count < 5
        if r == 2 and c == 2 and np.any(expected < 5):
            test_name = "Fisher's Exact Test"
            why_chosen = "2x2 table with small expected cell frequency (< 5)."
            statistic, p_val = stats.fisher_exact(ct.values)
        else:
            test_name = "Chi-Square Test of Independence"
            why_chosen = "Evaluates independence between two categorical variables."
            statistic = chi2

        # Cramer's V
        min_dim = min(r - 1, c - 1)
        cramers_v = math.sqrt(chi2 / (n_obs * min_dim)) if (n_obs > 0 and min_dim > 0) else 0.0
        effect_size = {
            "name": "cramers_v",
            "value": round(cramers_v, 4),
            "label": _label_effect_size("cramers_v", cramers_v),
        }

        significant = bool(p_val < alpha)
        if significant and effect_size["label"] == "negligible":
            warnings.append("Statistically significant association but negligible practical effect size.")
        warnings.append("Categorical association does not prove a causal relationship.")

        contingency_records = [
            {"category": str(idx), **{str(col): int(val) for col, val in row.items()}}
            for idx, row in ct.iterrows()
        ]

        return {
            "test_name": test_name,
            "why_chosen": why_chosen,
            "assumptions": {"expected_counts_ge_5": bool(np.all(expected >= 5))},
            "n": int(n_obs),
            "group_summary": contingency_records,
            "statistic": round(float(statistic), 4),
            "p_value": float(p_val),
            "alpha": alpha,
            "significant": significant,
            "effect_size": effect_size,
            "ci_95": None,
            "post_hoc": None,
            "warnings": warnings,
        }

    # -------------------------------------------------------------
    # Test Type: correlation (two numerics)
    # -------------------------------------------------------------
    elif req.test_type == "correlation":
        col2 = req.second_col or req.group_col
        if not col2:
            return {"error": "Correlation test requires 'second_col'."}

        if not pd.api.types.is_numeric_dtype(sub_df[req.target]) or not pd.api.types.is_numeric_dtype(sub_df[col2]):
            return {"error": f"Both columns ('{req.target}', '{col2}') must be numeric for correlation testing."}

        x = sub_df[req.target].values
        y = sub_df[col2].values
        n_obs = len(x)
        if n_obs < 5:
            return {"error": f"Insufficient sample size for correlation (n={n_obs} < 5)."}

        rng = np.random.RandomState(RANDOM_STATE)
        x_sub = rng.choice(x, size=min(n_obs, 5000), replace=False)
        y_sub = rng.choice(y, size=min(n_obs, 5000), replace=False)
        _, p_norm_x = stats.shapiro(x_sub)
        _, p_norm_y = stats.shapiro(y_sub)
        both_normal = (p_norm_x >= 0.05) and (p_norm_y >= 0.05)

        if both_normal:
            test_name = "Pearson Correlation"
            why_chosen = "Both variables pass normality checks (Shapiro p >= 0.05)."
            r_val, p_val = stats.pearsonr(x, y)
        else:
            test_name = "Spearman Rank Correlation"
            why_chosen = f"Non-normal distributions detected (Shapiro px={p_norm_x:.3g}, py={p_norm_y:.3g})."
            r_val, p_val = stats.spearmanr(x, y)

        effect_size = {
            "name": "r" if both_normal else "rho",
            "value": round(float(r_val), 4),
            "label": _label_effect_size("r", r_val),
        }

        significant = bool(p_val < alpha)
        warnings.append("Correlation does not imply causation in observational data.")
        if significant and effect_size["label"] == "negligible":
            warnings.append("Statistically significant correlation with negligible magnitude.")

        return {
            "test_name": test_name,
            "why_chosen": why_chosen,
            "assumptions": {"normality": both_normal},
            "n": int(n_obs),
            "group_summary": [
                {"variable": req.target, "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1))},
                {"variable": col2, "mean": float(np.mean(y)), "std": float(np.std(y, ddof=1))},
            ],
            "statistic": round(float(r_val), 4),
            "p_value": float(p_val),
            "alpha": alpha,
            "significant": significant,
            "effect_size": effect_size,
            "ci_95": None,
            "post_hoc": None,
            "warnings": warnings,
        }

    # -------------------------------------------------------------
    # Test Type: ab_proportion (binary target x 2-level group_col)
    # -------------------------------------------------------------
    elif req.test_type == "ab_proportion":
        group_col = req.group_col or req.second_col
        if not group_col:
            return {"error": "A/B proportion test requires 'group_col'."}

        # Coerce binary target
        target_s = sub_df[req.target]
        mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}
        coerced_target = target_s.astype(str).str.lower().map(mapping)
        if coerced_target.isna().any():
            return {"error": f"Target column '{req.target}' contains non-binary values (expected 0/1, bool, yes/no)."}

        groups = sub_df[group_col].astype(str)
        unique_groups = groups.unique()
        if len(unique_groups) != 2:
            return {"error": f"A/B proportion test requires exactly 2 groups in '{group_col}' (found {len(unique_groups)})."}

        g1, g2 = unique_groups[0], unique_groups[1]
        t1 = coerced_target[groups == g1].values
        t2 = coerced_target[groups == g2].values

        count1, nobs1 = int(np.sum(t1)), len(t1)
        count2, nobs2 = int(np.sum(t2)), len(t2)

        if nobs1 < 5 or nobs2 < 5:
            return {"error": f"Group sample sizes must be at least 5 (found {nobs1}, {nobs2})."}

        z_stat, p_val = proportions_ztest([count1, count2], [nobs1, nobs2])
        ci_low, ci_high = confint_proportions_2indep(count1, nobs1, count2, nobs2, method="wald", alpha=alpha)

        p1_rate = count1 / nobs1
        p2_rate = count2 / nobs2
        rate_diff = p1_rate - p2_rate

        effect_size = {
            "name": "risk_difference",
            "value": round(float(rate_diff), 4),
            "label": "large" if abs(rate_diff) > 0.1 else ("medium" if abs(rate_diff) > 0.05 else "small"),
        }

        significant = bool(p_val < alpha)
        if count1 < 5 or (nobs1 - count1) < 5 or count2 < 5 or (nobs2 - count2) < 5:
            warnings.append("Expected successes/failures in at least one group is < 5; z-test approximation may be strained.")

        return {
            "test_name": "Two-Proportion z-Test",
            "why_chosen": "Tests whether conversion/success rates between two independent variants differ significantly.",
            "assumptions": {"sample_size_adequate": bool(nobs1 >= 30 and nobs2 >= 30)},
            "n": int(nobs1 + nobs2),
            "group_summary": [
                {"group": str(g1), "n": nobs1, "successes": count1, "rate": round(p1_rate, 4)},
                {"group": str(g2), "n": nobs2, "successes": count2, "rate": round(p2_rate, 4)},
            ],
            "statistic": round(float(z_stat), 4),
            "p_value": float(p_val),
            "alpha": alpha,
            "significant": significant,
            "effect_size": effect_size,
            "ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
            "post_hoc": None,
            "warnings": warnings,
        }

    return {"error": f"Unknown test type '{req.test_type}'."}


def run_cohort_hypothesis_test(
    df: pd.DataFrame,
    *,
    ranking_column: str,
    outcome_column: Optional[str] = None,
    percentile: float = 5.0,
    direction: Literal["bottom", "top"] = "bottom",
    alpha: float = 0.05,
) -> dict:
    """Compare a requested percentile cohort with the remaining population.

    The ranking column defines the cohort. If an independent outcome is
    supplied, numeric outcomes use a group comparison, binary outcomes use a
    two-proportion test, and other categorical outcomes use a test of
    association. Without an outcome this returns the cohort definition only.
    """
    required = [ranking_column] + ([outcome_column] if outcome_column else [])
    if any(col not in df.columns for col in required):
        missing = [col for col in required if col not in df.columns]
        return {"error": f"Column(s) not found in dataset: {', '.join(missing)}."}
    if outcome_column and ranking_column == outcome_column:
        return {"error": "Choose a separate outcome column; testing the column used to rank the cohort would be circular."}
    if not 0 < percentile < 50:
        return {"error": "Percentile must be greater than 0 and less than 50."}
    if direction not in ("bottom", "top"):
        return {"error": "Cohort direction must be 'bottom' or 'top'."}
    if not pd.api.types.is_numeric_dtype(df[ranking_column]):
        return {"error": f"Ranking column '{ranking_column}' must be numeric."}

    ranked = df.loc[df[ranking_column].notna(), [ranking_column]].copy()
    if ranked.empty:
        return {"error": f"Ranking column '{ranking_column}' has no non-missing values."}

    quantile = percentile / 100 if direction == "bottom" else 1 - percentile / 100
    threshold = float(ranked[ranking_column].quantile(quantile))
    cohort_column = "__analysis_cohort__"
    while cohort_column in df.columns:
        cohort_column = f"_{cohort_column}_"
    if direction == "bottom":
        ranked[cohort_column] = np.where(ranked[ranking_column] <= threshold, f"Bottom {percentile:g}%", "Remaining")
    else:
        ranked[cohort_column] = np.where(ranked[ranking_column] >= threshold, f"Top {percentile:g}%", "Remaining")

    counts = ranked[cohort_column].value_counts().to_dict()
    if len(counts) != 2:
        return {"error": "The selected percentile did not produce two distinct cohorts; check the ranking column's ties and values."}
    cohort_label = f"{direction.title()} {percentile:g}%"
    actual_share = counts.get(cohort_label, 0) / len(ranked) * 100

    base_result = {
        "analysis": "cohort_hypothesis_test" if outcome_column else "cohort_definition",
        "ranking_column": ranking_column,
        "outcome_column": outcome_column,
        "cohort_direction": direction,
        "cohort_percentile": percentile,
        "ranking_threshold": threshold,
        "definition": (
            f"{cohort_label} means rows with {ranking_column} "
            f"{'<=' if direction == 'bottom' else '>='} {threshold:g}."
        ),
        "ranked_rows": int(len(ranked)),
        "cohort_counts": {str(key): int(value) for key, value in counts.items()},
        "actual_cohort_share_percent": round(actual_share, 2),
        "ties_may_expand_cohort": bool(not np.isclose(actual_share, percentile)),
        "selected_test_type": None,
        "test_result": None,
    }
    if not outcome_column:
        base_result["test_not_run_reason"] = (
            "No independent decision-relevant outcome column was available; the percentile cohort is defined, "
            "but this analysis cannot establish whether action on the cohort is justified."
        )
        return base_result

    valid = df.loc[df[ranking_column].notna() & df[outcome_column].notna(), [ranking_column, outcome_column]].copy()
    if len(valid) < 10:
        base_result["test_not_run_reason"] = f"At least 10 complete rows are required for comparison (found {len(valid)})."
        return base_result
    valid[cohort_column] = ranked[cohort_column].reindex(valid.index)
    valid = valid.dropna(subset=[cohort_column])
    comparison_counts = valid[cohort_column].value_counts().to_dict()
    if len(comparison_counts) != 2:
        base_result["test_not_run_reason"] = "Missing outcome values left too few observations in one cohort to compare."
        base_result["comparison_counts"] = {str(key): int(value) for key, value in comparison_counts.items()}
        return base_result
    outcome = valid[outcome_column]
    if pd.api.types.is_numeric_dtype(outcome):
        unique_values = set(outcome.dropna().unique().tolist())
        if len(unique_values) == 2 and unique_values.issubset({0, 1, 0.0, 1.0}):
            test_type = "ab_proportion"
        else:
            test_type = "compare_groups"
    else:
        test_type = "association"

    result = run_stat_test(
        valid,
        {
            "test_type": test_type,
            "target": outcome_column,
            "group_col": cohort_column,
            "second_col": cohort_column,
            "alpha": alpha,
        },
    )
    base_result["comparison_counts"] = {str(key): int(value) for key, value in comparison_counts.items()}
    if result.get("error"):
        base_result["test_not_run_reason"] = result["error"]
        return base_result
    base_result["selected_test_type"] = test_type
    base_result["test_result"] = result
    return base_result


# -------------------------------------------------------------
# ML / Forecasting
# -------------------------------------------------------------

class MLRequest(BaseModel):
    """Specification schema for supervised machine learning and time series forecasting."""

    target: str = Field(..., description="Target column to predict or forecast")
    features: Optional[list[str]] = Field(None, description="Explicit list of feature columns")
    task: Literal["auto", "regression", "classification", "forecast"] = Field(
        "auto", description="Task type: auto, regression, classification, or forecast"
    )
    time_col: Optional[str] = Field(None, description="Datetime column name for time series forecasting")
    horizon: int = Field(30, le=365, description="Forecast horizon steps")
    agg: Literal["sum", "mean"] = Field("sum", description="Aggregation function for forecasting periods")


def train_and_evaluate(df: pd.DataFrame, req: MLRequest | dict) -> dict:
    """Train, cross-validate, evaluate ML/forecast models, and save pipeline without data leakage."""
    if isinstance(req, dict):
        req = MLRequest(**req)

    if req.target not in df.columns:
        return {"error": f"Target column '{req.target}' not found in dataset."}

    # Infer Task
    task = req.task
    if task == "auto":
        if req.time_col is not None or "date" in req.target.lower():
            task = "forecast"
        elif pd.api.types.is_numeric_dtype(df[req.target]) and df[req.target].nunique(dropna=True) > 10:
            task = "regression"
        else:
            task = "classification"

    # =========================================================
    # TABULAR PATH (Regression / Classification)
    # =========================================================
    if task in ("regression", "classification"):
        # Drop rows with null target
        data = df.dropna(subset=[req.target]).copy()
        if len(data) < 30:
            return {"error": f"Insufficient data for supervised ML (only {len(data)} rows with non-null target)."}

        # Check classification classes
        if task == "classification":
            n_classes = data[req.target].nunique()
            if n_classes > 20:
                return {"error": f"Target has {n_classes} classes. Maximum allowed is 20."}
            if n_classes < 2:
                return {"error": "Classification requires at least 2 distinct classes in target."}

        # Select Features
        if req.features:
            features = [f for f in req.features if f in data.columns and f != req.target]
        else:
            features = []
            for col in data.columns:
                if col == req.target:
                    continue
                sem_type = data_tools._detect_semantic_type(data[col])
                if sem_type not in ("id_like", "text"):
                    features.append(col)

        if not features:
            return {"error": "No valid predictive features found."}

        # Expand datetime columns in features
        X_df = data[features].copy()
        for col in list(X_df.columns):
            if data_tools._detect_semantic_type(X_df[col]) == "datetime":
                parsed = pd.to_datetime(X_df[col], errors="coerce", format="mixed")
                X_df[f"{col}_year"] = parsed.dt.year.fillna(2000).astype(int)
                X_df[f"{col}_month"] = parsed.dt.month.fillna(1).astype(int)
                X_df[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(0).astype(int)
                X_df.drop(columns=[col], inplace=True)

        y = data[req.target]
        num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        cat_cols = [c for c in X_df.columns if c not in num_cols]

        # 80/20 train/test split (stratified for classification)
        stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
        )

        # Preprocessor Pipeline (FIT ON TRAIN ONLY -> NO LEAKAGE!)
        transformers = []
        if num_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, num_cols))

        if cat_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        # Models & Baselines
        if task == "regression":
            candidate_models = {
                "Ridge": Ridge(random_state=RANDOM_STATE),
                "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
                "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            }
            baseline_model = DummyRegressor(strategy="mean")
            scoring = "neg_root_mean_squared_error"
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        else:
            candidate_models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
                "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            }
            baseline_model = DummyClassifier(strategy="most_frequent")
            scoring = "f1_macro"
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        # 5-fold CV on train for model selection
        best_model_name = None
        best_cv_score = -float("inf")

        for name, clf in candidate_models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", clf)])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
            mean_score = float(np.mean(scores))
            if mean_score > best_cv_score:
                best_cv_score = mean_score
                best_model_name = name

        # Refit best model on full train and evaluate on test
        best_clf = candidate_models[best_model_name]
        best_pipeline = Pipeline([("prep", preprocessor), ("model", best_clf)])
        best_pipeline.fit(X_train, y_train)

        # Baseline evaluation on test
        base_pipeline = Pipeline([("prep", preprocessor), ("model", baseline_model)])
        base_pipeline.fit(X_train, y_train)
        y_base_pred = base_pipeline.predict(X_test)
        y_pred = best_pipeline.predict(X_test)

        warnings_list: list[str] = []
        metrics_dict: dict[str, Any] = {}

        if task == "regression":
            test_mae = float(mean_absolute_error(y_test, y_pred))
            test_rmse = float(root_mean_squared_error(y_test, y_pred))
            test_r2 = float(r2_score(y_test, y_pred))

            base_mae = float(mean_absolute_error(y_test, y_base_pred))
            base_rmse = float(root_mean_squared_error(y_test, y_base_pred))
            base_r2 = float(r2_score(y_test, y_base_pred))

            metrics_dict = {
                "model_name": best_model_name,
                "MAE": round(test_mae, 4),
                "RMSE": round(test_rmse, 4),
                "R2": round(test_r2, 4),
                "baseline_MAE": round(base_mae, 4),
                "baseline_RMSE": round(base_rmse, 4),
                "baseline_R2": round(base_r2, 4),
            }
            if test_r2 <= 0:
                warnings_list.append("Model R² is <= 0; it does not outperform a simple mean baseline on test data.")

            # Residuals plot
            fig_perf = px.scatter(
                x=y_test,
                y=y_pred,
                labels={"x": "Actual", "y": "Predicted"},
                title=f"{best_model_name}: Actual vs Predicted on Test Set",
                template="plotly_white",
            )
            fig_perf.add_trace(
                go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode="lines", name="Ideal")
            )

        else:  # classification
            test_acc = float(accuracy_score(y_test, y_pred))
            test_f1 = float(f1_score(y_test, y_pred, average="macro"))
            base_acc = float(accuracy_score(y_test, y_base_pred))
            base_f1 = float(f1_score(y_test, y_base_pred, average="macro"))

            cm = confusion_matrix(y_test, y_pred).tolist()

            roc_auc = None
            if len(np.unique(y_test)) == 2:
                try:
                    if hasattr(best_pipeline, "predict_proba"):
                        proba = best_pipeline.predict_proba(X_test)[:, 1]
                        roc_auc = float(roc_auc_score(y_test, proba))
                except Exception:
                    pass

            metrics_dict = {
                "model_name": best_model_name,
                "accuracy": round(test_acc, 4),
                "macro_F1": round(test_f1, 4),
                "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else None,
                "confusion_matrix": cm,
                "baseline_accuracy": round(base_acc, 4),
                "baseline_macro_F1": round(base_f1, 4),
            }
            if test_f1 <= base_f1:
                warnings_list.append("Model macro-F1 does not beat the majority-class Dummy baseline.")

            # Confusion matrix plot
            fig_perf = px.imshow(
                cm,
                text_auto=True,
                title=f"Confusion Matrix: {best_model_name}",
                labels=dict(x="Predicted", y="Actual"),
                template="plotly_white",
            )

        # Permutation Importance on test (top 10, n_repeats=5)
        perm = permutation_importance(
            best_pipeline, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring=scoring
        )
        importances = perm.importances_mean
        top_idx = np.argsort(importances)[::-1][:10]

        top_drivers = [
            {"feature": str(X_df.columns[i]), "importance": round(float(importances[i]), 4)}
            for i in top_idx
            if importances[i] > 0
        ]

        tables = {
            "metrics": pd.DataFrame([metrics_dict]),
            "top_drivers": pd.DataFrame(top_drivers) if top_drivers else pd.DataFrame(),
        }

        return {
            "task": task,
            "best_model": best_model_name,
            "metrics": metrics_dict,
            "top_drivers": top_drivers,
            "figure_json": pio.to_json(fig_perf),
            "tables": {k: data_tools.to_jsonable(v) for k, v in tables.items()},
            "warnings": warnings_list,
        }

    # =========================================================
    # TIME SERIES FORECAST PATH
    # =========================================================
    elif task == "forecast":
        time_col = req.time_col
        if not time_col:
            # Try to infer datetime column
            for col in df.columns:
                if data_tools._detect_semantic_type(df[col]) == "datetime":
                    time_col = col
                    break

        if not time_col:
            return {"error": "Time series forecast requires a datetime column ('time_col')."}

        # Parse time_col and sort
        ts_df = df[[time_col, req.target]].dropna().copy()
        ts_df[time_col] = pd.to_datetime(ts_df[time_col], errors="coerce", format="mixed")
        ts_df = ts_df.dropna().sort_values(by=time_col)

        if len(ts_df) < 10:
            return {"error": f"Insufficient non-null time series data points (found {len(ts_df)})."}

        # Infer frequency from median gap
        gaps = ts_df[time_col].diff().dropna()
        median_gap_days = gaps.median().total_seconds() / 86400.0

        if median_gap_days <= 1.5:
            freq = "D"
        elif median_gap_days <= 10:
            freq = "W"
        else:
            freq = "ME"

        # Aggregate per period
        agg_op = "sum" if req.agg == "sum" else "mean"
        ts_series = ts_df.set_index(time_col).resample(freq)[req.target].agg(agg_op)

        # Complete range & interpolate gaps (limit 3)
        ts_series = ts_series.interpolate(method="linear", limit=3).dropna()

        if len(ts_series) < 30:
            return {"error": f"At least 30 time periods required for forecasting (found {len(ts_series)})."}

        # Feature Engineering (Lags, rolling, calendar, trend) - NO LEAKAGE
        feat_df = pd.DataFrame({"y": ts_series})
        feat_df["lag_1"] = feat_df["y"].shift(1)
        feat_df["lag_2"] = feat_df["y"].shift(2)
        feat_df["lag_3"] = feat_df["y"].shift(3)

        seasonal_lag = 7 if freq == "D" else (4 if freq == "W" else (12 if (freq == "ME" and len(ts_series) >= 36) else None))
        if seasonal_lag:
            feat_df["lag_seasonal"] = feat_df["y"].shift(seasonal_lag)

        # Rolling mean(3) strictly on shifted values
        feat_df["rolling_mean_3"] = feat_df["y"].shift(1).rolling(3).mean()

        # Calendar features
        feat_df["dayofweek"] = feat_df.index.dayofweek
        feat_df["month"] = feat_df.index.month
        feat_df["trend"] = np.arange(len(feat_df))

        # Drop initial NA rows
        valid_feat = feat_df.dropna().copy()
        if len(valid_feat) < 20:
            return {"error": "Insufficient valid time periods after computing lag features."}

        # Chronological holdout = last 20% (min 8 periods)
        n_holdout = max(8, int(len(valid_feat) * 0.2))
        train_feat = valid_feat.iloc[:-n_holdout]
        holdout_feat = valid_feat.iloc[-n_holdout:]

        feature_cols = [c for c in valid_feat.columns if c != "y"]
        X_train, y_train = train_feat[feature_cols], train_feat["y"]
        X_holdout, y_holdout = holdout_feat[feature_cols], holdout_feat["y"]

        # Model selection via TimeSeriesSplit(3)
        tscv = TimeSeriesSplit(n_splits=3)
        candidate_models = {
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        }

        best_model_name = None
        best_cv_rmse = float("inf")

        for name, clf in candidate_models.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
            scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error")
            mean_rmse = -float(np.mean(scores))
            if mean_rmse < best_cv_rmse:
                best_cv_rmse = mean_rmse
                best_model_name = name

        # Evaluate best model on chronological holdout
        best_pipe = Pipeline([("scaler", StandardScaler()), ("model", candidate_models[best_model_name])])
        best_pipe.fit(X_train, y_train)
        y_holdout_pred = best_pipe.predict(X_holdout)

        holdout_mae = float(mean_absolute_error(y_holdout, y_holdout_pred))
        holdout_rmse = float(root_mean_squared_error(y_holdout, y_holdout_pred))

        # Baselines: Naive (last observed value) and Seasonal Naive
        # Naive: prediction is previous period's actual y
        naive_pred = holdout_feat["lag_1"].values
        naive_mae = float(mean_absolute_error(y_holdout, naive_pred))
        naive_rmse = float(root_mean_squared_error(y_holdout, naive_pred))

        skill = 1.0 - (holdout_mae / naive_mae) if naive_mae > 0 else 0.0

        warnings_list: list[str] = []
        if skill <= 0:
            warnings_list.append("Model does not beat the naive baseline; do not rely on this forecast.")

        # Refit best model on ALL data (train + holdout)
        X_all, y_all = valid_feat[feature_cols], valid_feat["y"]
        full_pipe = Pipeline([("scaler", StandardScaler()), ("model", candidate_models[best_model_name])])
        full_pipe.fit(X_all, y_all)

        # Recursive multi-step forecasting
        horizon = min(req.horizon, 365)
        residuals = y_holdout.values - y_holdout_pred
        resid_std = float(np.std(residuals)) if len(residuals) > 1 else 1.0

        forecast_records = []
        last_date = ts_series.index[-1]
        all_series = ts_series.copy()

        for step in range(1, horizon + 1):
            next_date = last_date + (pd.Timedelta(days=step) if freq == "D" else (pd.Timedelta(weeks=step) if freq == "W" else pd.DateOffset(months=step)))

            lag1 = float(all_series.iloc[-1])
            lag2 = float(all_series.iloc[-2]) if len(all_series) >= 2 else lag1
            lag3 = float(all_series.iloc[-3]) if len(all_series) >= 3 else lag2

            row_dict = {
                "lag_1": lag1,
                "lag_2": lag2,
                "lag_3": lag3,
                "rolling_mean_3": float(all_series.iloc[-3:].mean()),
                "dayofweek": next_date.dayofweek,
                "month": next_date.month,
                "trend": len(all_series),
            }
            if seasonal_lag:
                row_dict["lag_seasonal"] = (
                    float(all_series.iloc[-seasonal_lag])
                    if len(all_series) >= seasonal_lag
                    else lag1
                )

            X_next = pd.DataFrame([row_dict])[feature_cols]
            yhat = float(full_pipe.predict(X_next)[0])
            lo = round(yhat - 1.96 * resid_std, 2)
            hi = round(yhat + 1.96 * resid_std, 2)

            forecast_records.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "yhat": round(yhat, 2),
                "lo": lo,
                "hi": hi,
            })
            # Append recursive prediction
            all_series.loc[next_date] = yhat

        # Plotly Forecast Visualization with Confidence Interval Shading
        fig_fc = go.Figure()

        # History trace (last 60 points)
        recent_hist = ts_series.tail(60)
        fig_fc.add_trace(
            go.Scatter(
                x=[d.strftime("%Y-%m-%d") for d in recent_hist.index],
                y=recent_hist.values,
                mode="lines",
                name="Historical Actual",
                line=dict(color="#1f77b4", width=2),
            )
        )

        # Holdout Actual vs Predicted
        fig_fc.add_trace(
            go.Scatter(
                x=[d.strftime("%Y-%m-%d") for d in holdout_feat.index],
                y=y_holdout_pred,
                mode="lines+markers",
                name="Holdout Predicted",
                line=dict(color="#ff7f0e", dash="dash"),
            )
        )

        # Forecast line
        fc_dates = [r["date"] for r in forecast_records]
        fc_yhat = [r["yhat"] for r in forecast_records]
        fc_lo = [r["lo"] for r in forecast_records]
        fc_hi = [r["hi"] for r in forecast_records]

        fig_fc.add_trace(
            go.Scatter(
                x=fc_dates,
                y=fc_yhat,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#2ca02c", width=2),
            )
        )

        # Confidence Interval Band
        fig_fc.add_trace(
            go.Scatter(
                x=fc_dates + fc_dates[::-1],
                y=fc_hi + fc_lo[::-1],
                fill="toself",
                fillcolor="rgba(44, 160, 44, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Approximate 95% Interval",
            )
        )

        fig_fc.update_layout(
            title=f"{req.horizon}-Period Forecast for {req.target} ({best_model_name})",
            xaxis_title="Date",
            yaxis_title=req.target,
            template="plotly_white",
        )

        metrics_summary = {
            "model_name": best_model_name,
            "holdout_MAE": round(holdout_mae, 4),
            "holdout_RMSE": round(holdout_rmse, 4),
            "naive_MAE": round(naive_mae, 4),
            "naive_RMSE": round(naive_rmse, 4),
            "forecast_skill": round(skill, 4),
        }

        return {
            "task": "forecast",
            "best_model": best_model_name,
            "metrics": metrics_summary,
            "forecast_table": forecast_records,
            "figure_json": pio.to_json(fig_fc),
            "tables": {
                "forecast_summary": pd.DataFrame([metrics_summary]),
                "forecast_head": pd.DataFrame(forecast_records).head(10),
            },
            "warnings": warnings_list,
        }

    return {"error": f"Unsupported ML task '{task}'."}
