# Copyright 2025 Igor Bogdanov & Olga Manakina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import argparse
import os
import matplotlib
import re
from typing import List, Dict, Any, Tuple, Optional

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from typing import List, Dict, Any

try:
    from sklearn.metrics import cohen_kappa_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print(
        "Warning: scikit-learn not found. QWK calculations will be skipped. Please install with: pip install scikit-learn"
    )

METHOD_PREFIXES_GS = ["multi_ex", "multi_noex", "single_noex", "single_ex"]
MAB_LOG_DISTINCTIVE_COLUMNS = [
    "Step",
    "ChosenRecipe",
    "Epsilon",
]
GRID_LOG_DISTINCTIVE_COLUMNS = [f"{p}_PredictedScore" for p in METHOD_PREFIXES_GS]


def ensure_plot_dir(plot_dir: str):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created plot directory: {plot_dir}")


def format_approach_name(name_key: str):
    lk = name_key.lower()
    if "multistep+ex" == lk or "multi_ex" == lk:
        return "Multi-Step (Ex)"
    elif "multistep-noex" == lk or "multi_noex" == lk:
        return "Multi-Step (NoEx)"
    elif "singlestep+ex" == lk or "single_ex" == lk:
        return "Single-Step (Ex)"
    elif "singlestep-noex" == lk or "single_noex" == lk:
        return "Single-Step (NoEx)"
    else:
        name = name_key.replace("_", " ").replace("+", " ")
        name = name.replace("ex", "(Ex)").replace("noex", "(NoEx)")
        name = name.replace("multi", "Multi-Step").replace("single", "Single-Step")
        name_parts = [part.capitalize() for part in name.split(" ")]
        name = " ".join(name_parts)
        name = re.sub(r"\s*\(\s*", " (", name)
        name = re.sub(r"\s*\)\s*", ") ", name).strip()
        name = name.replace("( Ex )", "(Ex)").replace("( No Ex )", "(NoEx)")
        return name


def create_plot_title(
    base_title: str, provider: Optional[str], model: Optional[str]
) -> str:
    title = base_title
    if provider and model:
        title += f"\n(Model: {provider}/{model})"
    return title


def plot_bar_chart(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    filepath: str,
    provider: Optional[str],
    model: Optional[str],
    color="skyblue",
):
    plt.figure(figsize=(12, 7))
    data.plot(kind="bar", color=color)
    full_title = create_plot_title(title, provider, model)
    plt.title(full_title, fontsize=12)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")


def plot_grouped_bar_chart(
    df_compare: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    filepath: str,
    provider: Optional[str],
    model: Optional[str],
):
    plt.figure(figsize=(14, 8))
    df_compare.plot(
        kind="bar", x="Approach", y=[f"MAB_{metric_col}", f"Grid_{metric_col}"]
    )
    full_title = create_plot_title(title, provider, model)
    plt.title(full_title, fontsize=12)
    plt.ylabel(ylabel, fontsize=11)
    plt.xlabel("Approach", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(
        ["MAB", "Grid Search"],
        title="Experiment Type",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=9,
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0.08, 0.05, 0.90, 0.93])
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filepath: str,
    provider: Optional[str],
    model: Optional[str],
):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=label_col,
        style=label_col,
        s=120,
        palette="viridis",
    )
    full_title = create_plot_title(title, provider, model)
    plt.title(full_title, fontsize=12)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.legend(
        title="Approach", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")


def plot_mab_arm_pulls(
    arm_counts: pd.Series, plot_dir: str, provider: Optional[str], model: Optional[str]
):
    title = "MAB Arm Pull Distribution"
    filepath = os.path.join(plot_dir, "mab_arm_pulls.png")
    arm_counts.index = [format_approach_name(idx) for idx in arm_counts.index]
    plot_bar_chart(
        arm_counts,
        title,
        "Recipe (Arm)",
        "Total Pulls",
        filepath,
        provider,
        model,
        color="lightcoral",
    )


def plot_mab_avg_shaped_reward(
    arm_rewards: pd.Series, plot_dir: str, provider: Optional[str], model: Optional[str]
):
    title = "MAB Average Shaped Reward per Arm"
    filepath = os.path.join(plot_dir, "mab_avg_shaped_reward.png")
    arm_rewards.index = [format_approach_name(idx) for idx in arm_rewards.index]
    plot_bar_chart(
        arm_rewards,
        title,
        "Recipe (Arm)",
        "Average Shaped Reward",
        filepath,
        provider,
        model,
        color="mediumseagreen",
    )


def plot_mab_mae(
    arm_mae: pd.Series, plot_dir: str, provider: Optional[str], model: Optional[str]
):
    title = "MAB MAE per Arm (Accuracy)"
    filepath = os.path.join(plot_dir, "mab_mae.png")
    arm_mae.index = [format_approach_name(idx) for idx in arm_mae.index]
    plot_bar_chart(
        arm_mae,
        title,
        "Recipe (Arm)",
        "Mean Absolute Error",
        filepath,
        provider,
        model,
        color="cornflowerblue",
    )


def plot_mab_reward_over_time(
    log_df: pd.DataFrame,
    plot_dir: str,
    provider: Optional[str],
    model: Optional[str],
    window_size=50,
):
    plt.figure(figsize=(12, 7))
    for recipe in log_df["ChosenRecipe"].unique():
        recipe_df = log_df[log_df["ChosenRecipe"] == recipe]
        cumulative_avg_reward = recipe_df["Reward"].expanding().mean()
        plt.plot(
            recipe_df["Step"],
            cumulative_avg_reward,
            marker=".",
            linestyle="-",
            markersize=4,
            label=format_approach_name(recipe),
        )
    full_title = create_plot_title(
        "MAB: Cumulative Average Shaped Reward Over Steps", provider, model
    )
    plt.title(full_title, fontsize=12)
    plt.xlabel("Step", fontsize=11)
    plt.ylabel("Cumulative Average Shaped Reward", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    filepath = os.path.join(plot_dir, f"mab_avg_reward_over_time.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")


def plot_grid_metric_per_approach(
    data: pd.Series,
    metric_name: str,
    unit: str,
    plot_dir: str,
    provider: Optional[str],
    model: Optional[str],
    color="skyblue",
):
    title = f"Grid Search: Average {metric_name} per Approach"
    filepath = os.path.join(
        plot_dir, f"grid_avg_{metric_name.lower().replace(' ','_')}.png"
    )
    data.index = [format_approach_name(idx) for idx in data.index]
    plot_bar_chart(
        data,
        title,
        "Approach",
        f"Average {metric_name} {unit}",
        filepath,
        provider,
        model,
        color=color,
    )


def plot_grid_error_boxplot(
    log_df: pd.DataFrame, plot_dir: str, provider: Optional[str], model: Optional[str]
):
    plt.figure(figsize=(12, 7))
    error_data = []
    approach_names_formatted = []
    for prefix in METHOD_PREFIXES_GS:
        errors = log_df[f"{prefix}_err"].dropna()
        if not errors.empty:
            error_data.append(errors)
            approach_names_formatted.append(format_approach_name(prefix))

    if error_data:
        sns.boxplot(data=error_data)
        plt.xticks(
            ticks=range(len(approach_names_formatted)),
            labels=approach_names_formatted,
            rotation=45,
            ha="right",
            fontsize=10,
        )
        full_title = create_plot_title(
            "Grid Search: Distribution of Absolute Errors per Approach", provider, model
        )
        plt.title(full_title, fontsize=12)
        plt.ylabel("Absolute Error", fontsize=11)
        plt.xlabel("Approach", fontsize=11)
        plt.yticks(fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        filepath = os.path.join(plot_dir, "grid_error_distribution_boxplot.png")
        plt.savefig(filepath)
        print(f"Saved plot: {filepath}")
    else:
        print("Skipping error boxplot as no error data was available.")
    plt.close()


def plot_cumulative_tokens_mab(
    log_df: pd.DataFrame, plot_dir: str, provider: Optional[str], model: Optional[str]
):
    if "TotalTokens" not in log_df.columns or log_df["TotalTokens"].isnull().all():
        print(
            "Skipping MAB cumulative tokens plot due to missing or all NaN 'TotalTokens' data."
        )
        return
    plt.figure(figsize=(12, 7))
    log_df["CumulativeTotalTokens"] = log_df["TotalTokens"].cumsum()
    plt.plot(
        log_df["Step"],
        log_df["CumulativeTotalTokens"],
        marker=".",
        linestyle="-",
        markersize=4,
    )
    full_title = create_plot_title(
        "MAB: Cumulative Total Tokens Over Steps", provider, model
    )
    plt.title(full_title, fontsize=12)
    plt.xlabel("Step Number", fontsize=11)
    plt.ylabel("Cumulative Total Tokens", fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    filepath = os.path.join(plot_dir, "mab_cumulative_total_tokens.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")


def plot_cumulative_tokens_grid(
    log_df: pd.DataFrame, plot_dir: str, provider: Optional[str], model: Optional[str]
):
    agg_token_col = "Essay_Agg_TotalTokens"
    if agg_token_col not in log_df.columns or log_df[agg_token_col].isnull().all():
        print(
            f"Skipping Grid cumulative tokens plot due to missing or all NaN '{agg_token_col}' data."
        )
        return
    plt.figure(figsize=(12, 7))
    log_df["CumulativeEssayAggTokens"] = log_df[agg_token_col].cumsum()
    plt.plot(
        log_df.index,
        log_df["CumulativeEssayAggTokens"],
        marker=".",
        linestyle="-",
        markersize=4,
    )
    full_title = create_plot_title(
        "Grid Search: Cumulative Total Tokens Over Essays Processed", provider, model
    )
    plt.title(full_title, fontsize=12)
    plt.xlabel("Essay Processed Sequence (row index in log)", fontsize=11)
    plt.ylabel("Cumulative Total Tokens (all methods per essay)", fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    filepath = os.path.join(plot_dir, "grid_cumulative_total_tokens.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")


def plot_mab_cumulative_reward_per_arm(
    log_df: pd.DataFrame, plot_dir: str, provider: Optional[str], model: Optional[str]
):
    if (
        "ChosenRecipe" not in log_df.columns
        or "Reward" not in log_df.columns
        or "Step" not in log_df.columns
    ):
        print(
            "Skipping MAB cumulative reward per arm plot due to missing required columns."
        )
        return

    plt.figure(figsize=(12, 7))

    log_df["Step"] = log_df["Step"].astype(int)

    for recipe_name in log_df["ChosenRecipe"].unique():
        arm_df = log_df[log_df["ChosenRecipe"] == recipe_name].sort_values(by="Step")
        if not arm_df.empty:
            arm_df["CumulativeArmReward"] = arm_df["Reward"].cumsum()
            plt.plot(
                arm_df["Step"],
                arm_df["CumulativeArmReward"],
                marker=".",
                linestyle="-",
                markersize=4,
                label=format_approach_name(recipe_name),
            )

    full_title = create_plot_title(
        "MAB: Cumulative Shaped Reward per Arm Over Steps", provider, model
    )
    plt.title(full_title, fontsize=12)
    plt.xlabel("Step Number", fontsize=11)
    plt.ylabel("Cumulative Shaped Reward", fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    filepath = os.path.join(plot_dir, "mab_cumulative_reward_per_arm.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")


def plot_mab_projected_total_reward(
    arm_stats: pd.DataFrame,
    total_steps: int,
    plot_dir: str,
    provider: Optional[str],
    model: Optional[str],
):
    if "AvgShapedReward" not in arm_stats.columns:
        print(
            "Skipping MAB projected total reward plot due to missing 'AvgShapedReward'."
        )
        return

    if "ChosenRecipe" in arm_stats.columns:
        arm_stats_for_plot = arm_stats.set_index("ChosenRecipe").copy()
    else:
        print(
            "Warning: 'ChosenRecipe' column not found for projected reward plot. Cannot label arms correctly."
        )
        arm_stats_for_plot = arm_stats.copy()

    if "AvgShapedReward" not in arm_stats_for_plot.columns:
        print(
            "Skipping MAB projected total reward plot as 'AvgShapedReward' is missing after processing arm_stats."
        )
        return

    arm_stats_for_plot["ProjectedTotalReward"] = (
        arm_stats_for_plot["AvgShapedReward"] * total_steps
    )
    projected_rewards = arm_stats_for_plot["ProjectedTotalReward"].sort_values(
        ascending=False
    )

    title = f"MAB: Projected Total Shaped Reward (if each arm pulled for all {total_steps} steps)"
    filepath = os.path.join(plot_dir, "mab_projected_total_reward.png")
    projected_rewards.index = [
        format_approach_name(idx) for idx in projected_rewards.index
    ]
    plot_bar_chart(
        projected_rewards,
        title,
        "Recipe (Arm)",
        "Projected Total Shaped Reward",
        filepath,
        provider,
        model,
        color="lightsteelblue",
    )


def calculate_qwk_for_pairs(y_true_str: pd.Series, y_pred_str: pd.Series) -> float:
    if not SKLEARN_AVAILABLE:
        return np.nan

    possible_labels = [f"{s:.1f}" for s in np.arange(1.0, 9.5, 0.5)]

    valid_pairs = []
    for true_s, pred_s in zip(y_true_str, y_pred_str):
        true_val, pred_val = None, None
        try:
            h_float = float(true_s)
            if 1.0 <= h_float <= 9.0 and h_float * 10 % 5 == 0:
                true_val = f"{h_float:.1f}"
        except (ValueError, TypeError):
            pass

        try:
            p_float = float(pred_s)
            if 1.0 <= p_float <= 9.0 and p_float * 10 % 5 == 0:
                pred_val = f"{p_float:.1f}"
        except (ValueError, TypeError):
            pass

        if true_val is not None and pred_val is not None:
            valid_pairs.append((true_val, pred_val))

    if len(valid_pairs) < 2:
        return np.nan

    y_true_actual = [p[0] for p in valid_pairs]
    y_pred_actual = [p[1] for p in valid_pairs]

    try:
        current_unique_scores = sorted(list(set(y_true_actual + y_pred_actual)))
        final_labels = sorted(list(set(possible_labels + current_unique_scores)))

        return cohen_kappa_score(
            y_true_actual, y_pred_actual, labels=final_labels, weights="quadratic"
        )
    except Exception as e:
        print(f"Error calculating QWK: {e}")
        return np.nan


def analyze_mab_log(
    log_df: pd.DataFrame,
    log_filepath: str,
    plot_dir: str,
    provider_name: Optional[str],
    model_name: Optional[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    print(f"\n--- MAB Log Analysis: {log_filepath} ---")
    ensure_plot_dir(plot_dir)

    numeric_cols = [
        "PromptTokens",
        "CompletionTokens",
        "TotalTokens",
        "Latency",
        "EstimatedCost",
        "Reward",
    ]
    for col in numeric_cols:
        if col in log_df.columns:
            log_df[col] = pd.to_numeric(log_df[col], errors="coerce")
        else:
            print(
                f"Warning: Expected column '{col}' not found in MAB log for numeric conversion."
            )
            return
    total_prompt_tokens_mab = log_df["PromptTokens"].sum(skipna=True)
    total_completion_tokens_mab = log_df["CompletionTokens"].sum(skipna=True)
    total_tokens_logged_mab = log_df["TotalTokens"].sum(skipna=True)
    total_latency_mab = log_df["Latency"].sum(skipna=True)
    total_estimated_cost_mab = log_df["EstimatedCost"].sum(skipna=True)
    num_mab_steps_logged = len(log_df)

    print("\nOverall MAB Experiment Totals (Actual Operations):")
    print(f"  MAB Steps Logged:        {num_mab_steps_logged}")
    print(f"  Total Prompt Tokens:     {total_prompt_tokens_mab:,.0f}")
    print(f"  Total Completion Tokens: {total_completion_tokens_mab:,.0f}")
    print(f"  Total Tokens:            {total_tokens_logged_mab:,.0f}")
    print(f"  Total Latency (sum):     {total_latency_mab:.2f} seconds")
    print(f"  Total Estimated Cost:    ${total_estimated_cost_mab:.4f}")

    def calculate_accuracy_error(row):
        try:
            pred_score_str = str(row["PredictedScore"])
            human_score_str = str(row["HumanScore"])
            if not pred_score_str or pred_score_str.lower() == "nan":
                return np.nan
            pred = float(pred_score_str)
            human = float(human_score_str)
            if not (1.0 <= pred <= 9.0 and pred * 10 % 5 == 0):
                return np.nan
            if not (1.0 <= human <= 9.0 and human * 10 % 5 == 0):
                return np.nan
            return abs(pred - human)
        except (ValueError, TypeError):
            return np.nan

    log_df["AccuracyError"] = log_df.apply(calculate_accuracy_error, axis=1)

    print("\nPer-Arm Statistics (from MAB Log):")
    arm_stats = (
        log_df.groupby("ChosenRecipe")
        .agg(
            PullCount=("ChosenRecipe", "size"),
            MAE=("AccuracyError", "mean"),
            AvgShapedReward=("Reward", "mean"),
            AvgEstimatedCost=("EstimatedCost", "mean"),
            AvgPromptTokens=("PromptTokens", "mean"),
            AvgCompletionTokens=("CompletionTokens", "mean"),
            AvgTotalTokens=("TotalTokens", "mean"),
            AvgLatency=("Latency", "mean"),
        )
        .reset_index()
    )
    qwk_scores = {}
    if SKLEARN_AVAILABLE:
        for recipe_name in arm_stats["ChosenRecipe"].unique():
            arm_data = log_df[log_df["ChosenRecipe"] == recipe_name]
            qwk_scores[recipe_name] = calculate_qwk_for_pairs(
                arm_data["HumanScore"], arm_data["PredictedScore"]
            )
        arm_stats["QWK"] = arm_stats["ChosenRecipe"].map(qwk_scores)
    else:
        arm_stats["QWK"] = np.nan

    mab_display_cols = [
        "PullCount",
        "MAE",
        "QWK",
        "AvgShapedReward",
        "AvgPromptTokens",
        "AvgCompletionTokens",
        "AvgTotalTokens",
        "AvgLatency",
        "AvgEstimatedCost",
    ]
    for col in mab_display_cols:
        if col not in arm_stats.columns and col != "PullCount":
            arm_stats[col] = 0.0
    arm_stats = arm_stats[["ChosenRecipe"] + mab_display_cols]
    arm_stats.sort_values(
        by=["MAE", "AvgShapedReward"], ascending=[True, False], inplace=True
    )

    header_approach = "Approach (Arm)"
    header_pulls = "Pulls"
    header_mae = "MAE"
    header_avg_shaped_reward = "AvgShapedRew"
    header_avg_prompt_tk = "AvgPromptTk"
    header_avg_compl_tk = "AvgComplTk"
    header_avg_total_tk = "AvgTotalTk"
    header_avg_latency = "AvgLatency"
    header_avg_cost = "AvgCost"
    header_qwk = "QWK"

    print(
        f"{header_approach:<25} {header_pulls:>7} {header_mae:>7} {header_qwk:>7} {header_avg_shaped_reward:>12} "
        f"{header_avg_prompt_tk:>12} {header_avg_compl_tk:>12} {header_avg_total_tk:>12} "
        f"{header_avg_latency:>11} {header_avg_cost:>15}"
    )
    print("-" * 140)

    for _, row_data in arm_stats.iterrows():
        approach_name_formatted = format_approach_name(row_data["ChosenRecipe"])
        print(
            f"{approach_name_formatted:<25} "
            f"{int(row_data['PullCount']):>7} "
            f"{row_data['MAE']:>7.3f} "
            f"{row_data['QWK']:>7.3f} "
            f"{row_data['AvgShapedReward']:>12.4f} "
            f"{row_data['AvgPromptTokens']:>12,.1f} "
            f"{row_data['AvgCompletionTokens']:>12,.1f} "
            f"{row_data['AvgTotalTokens']:>12,.1f} "
            f"{row_data['AvgLatency']:>10.3f}s "
            f"{row_data['AvgEstimatedCost']:>15.8f}"
        )

    if not arm_stats.empty:
        plot_mab_arm_pulls(
            arm_stats.set_index("ChosenRecipe")["PullCount"],
            plot_dir,
            provider_name,
            model_name,
        )
        plot_mab_avg_shaped_reward(
            arm_stats.set_index("ChosenRecipe")["AvgShapedReward"],
            plot_dir,
            provider_name,
            model_name,
        )
        plot_mab_mae(
            arm_stats.set_index("ChosenRecipe")["MAE"],
            plot_dir,
            provider_name,
            model_name,
        )

        plot_data_scatter = arm_stats.rename(columns={"ChosenRecipe": "Approach"})
        plot_data_scatter["Approach"] = plot_data_scatter["Approach"].apply(
            format_approach_name
        )

        plot_scatter(
            plot_data_scatter,
            "AvgTotalTokens",
            "MAE",
            "Approach",
            "MAB: MAE vs. Avg Total Tokens per Arm",
            "Avg Total Tokens",
            "MAE",
            os.path.join(plot_dir, "mab_mae_vs_tokens.png"),
            provider_name,
            model_name,
        )
        plot_scatter(
            plot_data_scatter,
            "AvgEstimatedCost",
            "MAE",
            "Approach",
            "MAB: MAE vs. Avg Estimated Cost per Arm",
            "Avg Estimated Cost ($)",
            "MAE",
            os.path.join(plot_dir, "mab_mae_vs_cost.png"),
            provider_name,
            model_name,
        )
    plot_mab_reward_over_time(log_df, plot_dir, provider_name, model_name)
    plot_cumulative_tokens_mab(log_df, plot_dir, provider_name, model_name)
    plot_mab_cumulative_reward_per_arm(log_df, plot_dir, provider_name, model_name)
    if not arm_stats.empty:
        plot_mab_projected_total_reward(
            arm_stats.copy(), num_mab_steps_logged, plot_dir, provider_name, model_name
        )

    if not log_df.empty:
        plot_mab_reward_over_time(log_df, plot_dir, provider_name, model_name)

    overall_mab_totals = {
        "Type": "MAB",
        "StepsLogged": num_mab_steps_logged,
        "TotalPromptTokens": total_prompt_tokens_mab,
        "TotalCompletionTokens": total_completion_tokens_mab,
        "TotalTokens": total_tokens_logged_mab,
        "TotalLatency": total_latency_mab,
        "TotalEstimatedCost": total_estimated_cost_mab,
    }
    return arm_stats, pd.Series(overall_mab_totals)


def analyze_grid_log(
    log_df: pd.DataFrame,
    log_filepath: str,
    plot_dir: str,
    provider_name: Optional[str],
    model_name: Optional[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    print(f"\n--- Grid Search Log Analysis: {log_filepath} ---")
    ensure_plot_dir(plot_dir)

    for prefix in METHOD_PREFIXES_GS:
        if f"{prefix}_err" in log_df.columns:
            log_df[f"{prefix}_err"] = pd.to_numeric(
                log_df[f"{prefix}_err"], errors="coerce"
            )
        else:
            log_df[f"{prefix}_err"] = np.nan
        for metric_suffix in [
            "PromptTokens",
            "CompletionTokens",
            "TotalTokens",
            "Latency",
            "EstimatedCost",
        ]:
            col_name = f"{prefix}_{metric_suffix}"
            if col_name in log_df.columns:
                log_df[col_name] = pd.to_numeric(log_df[col_name], errors="coerce")
            else:
                log_df[col_name] = 0

    for metric_suffix in [
        "PromptTokens",
        "CompletionTokens",
        "TotalTokens",
        "Latency",
        "EstimatedCost",
    ]:
        col_name = f"Essay_Agg_{metric_suffix}"
        if col_name in log_df.columns:
            log_df[col_name] = pd.to_numeric(log_df[col_name], errors="coerce")
        else:
            print(f"Warning: Expected column '{col_name}' not found in Grid log.")

    print("\nFinal Mean Absolute Errors (MAE) per Approach:")
    approach_maes = {}
    approach_qwks = {}

    for prefix in METHOD_PREFIXES_GS:
        mae = log_df[f"{prefix}_err"].mean()
        approach_maes[prefix] = mae
        print(f"{format_approach_name(prefix):<30} MAE: {mae:.3f}")

        if SKLEARN_AVAILABLE:
            true_scores_col = "HumanScore"
            pred_scores_col = f"{prefix}_PredictedScore"
            if true_scores_col in log_df.columns and pred_scores_col in log_df.columns:
                approach_qwks[prefix] = calculate_qwk_for_pairs(
                    log_df[true_scores_col], log_df[pred_scores_col]
                )
                print(
                    f"{format_approach_name(prefix):<30} QWK: {approach_qwks[prefix]:.3f}"
                )
            else:
                print(
                    f"Skipping QWK for {format_approach_name(prefix)} due to missing score columns."
                )
                approach_qwks[prefix] = np.nan
        else:
            approach_qwks[prefix] = np.nan

    print("\nAverage Usage Statistics per Approach (from log file):")
    agg_metrics_from_log = {}
    grid_summary_list = []
    for prefix in METHOD_PREFIXES_GS:
        current_metrics = {
            "Approach": format_approach_name(prefix),
            "MAE": approach_maes.get(prefix, np.nan),
            "QWK": approach_qwks.get(prefix, np.nan),
            "AvgPromptTokens": log_df[f"{prefix}_PromptTokens"].mean(),
            "AvgCompletionTokens": log_df[f"{prefix}_CompletionTokens"].mean(),
            "AvgTotalTokens": log_df[f"{prefix}_TotalTokens"].mean(),
            "AvgLatency": log_df[f"{prefix}_Latency"].mean(),
            "AvgEstimatedCost": log_df[f"{prefix}_EstimatedCost"].mean(),
        }
        grid_summary_list.append(current_metrics)
    grid_summary_df = pd.DataFrame(grid_summary_list).set_index("Approach")

    header_approach = "Approach"
    header_mae_col = "MAE"
    header_qwk = "QWK"
    header_prompt_tk = "AvgPromptTk"
    header_compl_tk = "AvgComplTk"
    header_total_tk = "AvgTotalTk"
    header_latency = "AvgLatency"
    header_cost = "AvgCost"

    print(
        f"{header_approach:<30} {header_mae_col:>7} {header_qwk:>7} {header_prompt_tk:>12} {header_compl_tk:>12} {header_total_tk:>12} {header_latency:>11} {header_cost:>15}"
    )
    print("-" * 115)
    for index, row_data in grid_summary_df.iterrows():
        print(
            (
                f"{index:<30} "
                f"{row_data['MAE']:>7.3f} "
                f"{row_data['QWK']:>7.3f} "
                f"{row_data['AvgPromptTokens']:>12,.1f} "
                f"{row_data['AvgCompletionTokens']:>12,.1f} "
                f"{row_data['AvgTotalTokens']:>12,.1f} "
                f"{row_data['AvgLatency']:>10.3f}s "
                f"{row_data['AvgEstimatedCost']:>15.8f}"
            )
        )

    if "Essay_Agg_TotalTokens" in log_df.columns:
        total_prompt_tokens_run = log_df["Essay_Agg_PromptTokens"].sum()
        total_completion_tokens_run = log_df["Essay_Agg_CompletionTokens"].sum()
        total_tokens_run = log_df["Essay_Agg_TotalTokens"].sum()
        total_latency_run = log_df["Essay_Agg_TotalLatency"].sum()
        total_cost_run = log_df["Essay_Agg_EstimatedCost"].sum()
        num_processed_essays = len(log_df)

        print(
            "\nOverall Grid Search Usage Statistics (Sum of per-essay totals from log):"
        )
        print(f"  Essays Processed:        {num_processed_essays}")
        print(f"  Total Prompt Tokens:     {total_prompt_tokens_run:,.0f}")
        print(f"  Total Completion Tokens: {total_completion_tokens_run:,.0f}")
        print(f"  Total Tokens:            {total_tokens_run:,.0f}")
        print(f"  Total Latency (sum):     {total_latency_run:.2f} seconds")
        print(f"  Total Estimated Cost:    ${total_cost_run:.4f}")
    else:
        print(
            "\nOverall Grid Search Usage Statistics could not be calculated (missing Essay_Agg_ columns)."
        )
        total_prompt_tokens_run = 0
        total_completion_tokens_run = 0
        total_tokens_run = 0
        total_latency_run = 0.0
        total_cost_run = 0.0
        num_processed_essays = len(log_df) if not log_df.empty else 0

    if not grid_summary_df.empty:
        plot_grid_metric_per_approach(
            grid_summary_df["MAE"],
            "MAE",
            "",
            plot_dir,
            provider_name,
            model_name,
            color="cornflowerblue",
        )
        plot_grid_metric_per_approach(
            grid_summary_df["AvgTotalTokens"],
            "Total Tokens",
            "tokens",
            plot_dir,
            provider_name,
            model_name,
            color="lightsalmon",
        )
        plot_grid_metric_per_approach(
            grid_summary_df["AvgEstimatedCost"],
            "Estimated Cost",
            "($)",
            plot_dir,
            provider_name,
            model_name,
            color="lightgreen",
        )
        latency_data_for_plot = pd.Series(
            {
                format_approach_name(p): log_df[f"{p}_Latency"].mean()
                for p in METHOD_PREFIXES_GS
            }
        )
        plot_grid_metric_per_approach(
            latency_data_for_plot,
            "Latency",
            "(s)",
            plot_dir,
            provider_name,
            model_name,
            color="gold",
        )
        plot_grid_error_boxplot(log_df, plot_dir, provider_name, model_name)
        plot_scatter(
            grid_summary_df.reset_index(),
            "AvgTotalTokens",
            "MAE",
            "Approach",
            "Grid Search: MAE vs. Avg Total Tokens",
            "Avg Total Tokens",
            "MAE",
            os.path.join(plot_dir, "grid_mae_vs_tokens.png"),
            provider_name,
            model_name,
        )
        plot_scatter(
            grid_summary_df.reset_index(),
            "AvgEstimatedCost",
            "MAE",
            "Approach",
            "Grid Search: MAE vs. Avg Estimated Cost",
            "Avg Estimated Cost ($)",
            "MAE",
            os.path.join(plot_dir, "grid_mae_vs_cost.png"),
            provider_name,
            model_name,
        )
    plot_cumulative_tokens_grid(log_df, plot_dir, provider_name, model_name)

    overall_grid_totals = {
        "Type": "GridSearch",
        "EssaysProcessed": num_processed_essays,
        "TotalPromptTokens": total_prompt_tokens_run,
        "TotalCompletionTokens": total_completion_tokens_run,
        "TotalTokens": total_tokens_run,
        "TotalLatency": total_latency_run,
        "TotalEstimatedCost": total_cost_run,
    }
    return grid_summary_df, pd.Series(overall_grid_totals)


def plot_overall_spending_comparison(
    mab_totals: pd.Series,
    grid_totals: pd.Series,
    plot_dir: str,
    provider: Optional[str],
    model: Optional[str],
):
    if mab_totals is None or grid_totals is None:
        print(
            "Skipping overall spending comparison plots, missing data for MAB or Grid."
        )
        return

    metrics_to_plot = {
        "TotalEstimatedCost": "Total Estimated Cost ($)",
        "TotalTokens": "Total Tokens",
        "TotalPromptTokens": "Total Prompt Tokens",
        "TotalCompletionTokens": "Total Completion Tokens",
    }

    for metric_key, plot_ylabel in metrics_to_plot.items():
        mab_value = mab_totals.get(metric_key, 0)
        grid_value = grid_totals.get(metric_key, 0)

        if pd.isna(mab_value) and pd.isna(grid_value):
            print(
                f"Skipping plot for {plot_ylabel} as both MAB and Grid values are NaN or missing."
            )
            continue

        mab_value = 0 if pd.isna(mab_value) else mab_value
        grid_value = 0 if pd.isna(grid_value) else grid_value

        data_to_plot = pd.Series(
            [mab_value, grid_value], index=["MAB Run", "Grid Search Run"]
        )

        plt.figure(figsize=(8, 6))
        data_to_plot.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
        full_title = create_plot_title(
            f"Overall Experiment Comparison: {plot_ylabel}", provider, model
        )
        plt.title(full_title, fontsize=12)
        plt.ylabel(plot_ylabel, fontsize=11)
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        filepath = os.path.join(
            plot_dir, f"overall_comparison_{metric_key.lower()}.png"
        )
        plt.savefig(filepath)
        plt.close()
        print(f"Saved plot: {filepath}")


def plot_cumulative_tokens_comparison(
    mab_log_df: pd.DataFrame,
    grid_log_df: pd.DataFrame,
    plot_dir: str,
    provider: Optional[str],
    model: Optional[str],
):
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()

    if (
        mab_log_df is not None
        and "TotalTokens" in mab_log_df.columns
        and not mab_log_df["TotalTokens"].isnull().all()
    ):
        mab_log_df_sorted = mab_log_df.sort_values(by="Step")
        mab_log_df_sorted["CumulativeTotalTokens"] = mab_log_df_sorted[
            "TotalTokens"
        ].cumsum()
        ax1.plot(
            mab_log_df_sorted["Step"],
            mab_log_df_sorted["CumulativeTotalTokens"],
            marker=".",
            linestyle="-",
            markersize=4,
            color="#1f77b4",
            label="MAB: Cumulative Tokens (per step)",
        )
    else:
        print("Skipping MAB cumulative tokens in comparison plot due to missing data.")

    agg_token_col_grid = "Essay_Agg_TotalTokens"
    if (
        grid_log_df is not None
        and agg_token_col_grid in grid_log_df.columns
        and not grid_log_df[agg_token_col_grid].isnull().all()
    ):
        grid_log_df["CumulativeEssayAggTokens"] = grid_log_df[
            agg_token_col_grid
        ].cumsum()
        ax1.plot(
            grid_log_df.index,
            grid_log_df["CumulativeEssayAggTokens"],
            marker="x",
            linestyle="--",
            markersize=4,
            color="#ff7f0e",
            label="Grid: Cumulative Tokens (per essay, all methods)",
        )
    else:
        print("Skipping Grid cumulative tokens in comparison plot due to missing data.")

    full_title_comp = create_plot_title(
        "Comparison: Cumulative Token Spending Growth", provider, model
    )
    ax1.set_title(full_title_comp, fontsize=12)
    ax1.set_xlabel("MAB Steps / Grid Essays Processed (Sequence Index)", fontsize=11)
    ax1.set_ylabel("Cumulative Total Tokens", fontsize=11)
    ax1.tick_params(axis="y", labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc="upper left", fontsize=10)

    ax1.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    filepath = os.path.join(plot_dir, "comparison_cumulative_tokens_single_y.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")


def analyze_comparison(
    mab_summary: pd.DataFrame,
    grid_summary: pd.DataFrame,
    mab_overall_totals: pd.Series,
    grid_overall_totals: pd.Series,
    plot_dir: str,
    mab_log_df_full: pd.DataFrame,
    grid_log_df_full: pd.DataFrame,
    provider_name: Optional[str],
    model_name: Optional[str],
):
    print("\n--- MAB vs. Grid Search Comparison ---")
    ensure_plot_dir(plot_dir)

    print("\nDEBUG: mab_summary at start of analyze_comparison:")
    print(mab_summary.to_string())
    print(f"MAB Summary Index: {mab_summary.index.names}")
    print(f"MAB Summary Columns: {mab_summary.columns.tolist()}")

    print("\nDEBUG: grid_summary at start of analyze_comparison:")
    print(grid_summary.to_string())
    print(f"Grid Summary Index: {grid_summary.index.names}")
    print(f"Grid Summary Columns: {grid_summary.columns.tolist()}")

    if "ChosenRecipe" in mab_summary.index.names:
        mab_summary = mab_summary.reset_index()

    mab_summary.rename(columns={"ChosenRecipe": "Approach"}, inplace=True)
    mab_summary["Approach"] = mab_summary["Approach"].apply(format_approach_name)
    mab_summary.set_index("Approach", inplace=True)

    comparison_metrics = {
        "MAE": ("MAE", "MAE (Lower is Better)", True),
        "QWK": ("QWK", "Quadratic Weighted Kappa (Higher is Better)", False),
        "AvgTotalTokens": ("AvgTotalTokens", "Average Total Tokens", False),
        "AvgEstimatedCost": ("AvgEstimatedCost", "Average Estimated Cost ($)", True),
        "AvgLatency": ("AvgLatency", "Average Latency (s)", True),
    }

    for metric_key, (
        col_name,
        title_metric,
        lower_is_better,
    ) in comparison_metrics.items():
        comparison_data = []
        for approach_prefix in METHOD_PREFIXES_GS:
            formatted_name = format_approach_name(approach_prefix)

            mab_val = (
                mab_summary.loc[formatted_name, col_name]
                if formatted_name in mab_summary.index
                and col_name in mab_summary.columns
                else np.nan
            )
            grid_val = (
                grid_summary.loc[formatted_name, col_name]
                if formatted_name in grid_summary.index
                and col_name in grid_summary.columns
                else np.nan
            )

            comparison_data.append(
                {
                    "Approach": formatted_name,
                    f"MAB_{col_name}": mab_val,
                    f"Grid_{col_name}": grid_val,
                }
            )

        df_compare = pd.DataFrame(comparison_data)
        if (
            df_compare.empty
            or df_compare[[f"MAB_{col_name}", f"Grid_{col_name}"]].isnull().all().all()
        ):
            print(f"Skipping comparison plot for {title_metric} due to missing data.")
            continue

        plot_title = f"Comparison: {title_metric}"
        plot_filepath = os.path.join(plot_dir, f"comparison_{metric_key.lower()}.png")
        plot_grouped_bar_chart(
            df_compare,
            col_name,
            plot_title,
            title_metric,
            plot_filepath,
            provider_name,
            model_name,
        )

    MAX_POSSIBLE_ERROR = 8.0
    accuracy_comparison_data = []
    for approach_prefix in METHOD_PREFIXES_GS:
        formatted_name = format_approach_name(approach_prefix)

        mab_mae = (
            mab_summary.loc[formatted_name, "MAE"]
            if formatted_name in mab_summary.index and "MAE" in mab_summary.columns
            else np.nan
        )
        grid_mae = (
            grid_summary.loc[formatted_name, "MAE"]
            if formatted_name in grid_summary.index and "MAE" in grid_summary.columns
            else np.nan
        )

        mab_qwk = (
            mab_summary.loc[formatted_name, "QWK"]
            if formatted_name in mab_summary.index and "QWK" in mab_summary.columns
            else np.nan
        )
        grid_qwk = (
            grid_summary.loc[formatted_name, "QWK"]
            if formatted_name in grid_summary.index and "QWK" in grid_summary.columns
            else np.nan
        )

        mab_perf_score = (
            (MAX_POSSIBLE_ERROR - mab_mae) if not pd.isna(mab_mae) else np.nan
        )
        grid_perf_score = (
            (MAX_POSSIBLE_ERROR - grid_mae) if not pd.isna(grid_mae) else np.nan
        )

        accuracy_comparison_data.append(
            {
                "Approach": formatted_name,
                "MAB_AccuracyPerf": mab_perf_score,
                "Grid_AccuracyPerf": grid_perf_score,
                "MAB_QWK": mab_qwk,
                "Grid_QWK": grid_qwk,
            }
        )
    df_accuracy_compare = pd.DataFrame(accuracy_comparison_data)

    if not (
        df_accuracy_compare.empty
        or df_accuracy_compare[["MAB_AccuracyPerf", "Grid_AccuracyPerf"]]
        .isnull()
        .all()
        .all()
    ):
        plot_grouped_bar_chart(
            df_accuracy_compare,
            "AccuracyPerf",
            "Comparison: Accuracy Performance Score (Higher is Better)",
            f"Accuracy Score ({MAX_POSSIBLE_ERROR:.1f} - MAE)",
            os.path.join(plot_dir, "comparison_accuracy_performance.png"),
            provider_name,
            model_name,
        )
    else:
        print("Skipping Accuracy Performance comparison plot due to missing MAE data.")

    if not (
        df_accuracy_compare.empty
        or df_accuracy_compare[["MAB_QWK", "Grid_QWK"]].isnull().all().all()
    ):
        plot_grouped_bar_chart(
            df_accuracy_compare,
            "QWK",
            "Comparison: Quadratic Weighted Kappa (QWK - Higher is Better)",
            "QWK Score",
            os.path.join(plot_dir, "comparison_qwk.png"),
            provider_name,
            model_name,
        )
    else:
        print("Skipping QWK comparison plot due to missing QWK data.")

    if mab_overall_totals is not None and grid_overall_totals is not None:
        plot_overall_spending_comparison(
            mab_overall_totals, grid_overall_totals, plot_dir, provider_name, model_name
        )

    if mab_log_df_full is not None and grid_log_df_full is not None:
        plot_cumulative_tokens_comparison(
            mab_log_df_full, grid_log_df_full, plot_dir, provider_name, model_name
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MAB or Grid Search experiment log files."
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        type=str,
        help="Path(s) to the experiment log CSV file(s) to analyze (1 or 2 files). ORDER: MAB log then Grid log if providing two.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save plots. If None, a directory is created based on the first log file name.",
    )
    args = parser.parse_args()

    if not args.log_files:
        print("No log files provided. Exiting.")
        return

    dfs = {}
    log_filepaths = {}
    provider_model_info = {
        "provider": None,
        "model": None,
    }
    for i, log_file_path in enumerate(args.log_files):
        if not os.path.exists(log_file_path):
            print(f"Error: Log file not found at {log_file_path}")
            continue
        try:
            df = pd.read_csv(log_file_path)
            print(f"Successfully loaded log file: {log_file_path} with {len(df)} rows.")

            if i == 0:
                filename = os.path.basename(log_file_path)
                patterns_to_try = [
                    r"(?:mab_log|grid_log)_([^_]+)_(.+?)_(?:\d+s(?:eps){0,1}|allessays)",
                    r"(?:mab_log|grid_log)_([^_]+)_(.+?)_\d{8}_\d{6}",
                ]
                found_match = False
                for idx, pattern in enumerate(patterns_to_try):
                    match = re.search(pattern, filename)
                    if match:
                        provider_model_info["provider"] = match.group(1)
                        provider_model_info["model"] = match.group(2).replace(
                            ".csv", ""
                        )
                        found_match = True
                        break

                if provider_model_info["provider"] and provider_model_info["model"]:
                    print(
                        f"  Inferred Provider: {provider_model_info['provider']}, Model: {provider_model_info['model']}"
                    )
                else:
                    print(
                        f"  Could not infer provider/model from filename '{filename}'. Subtitles will be generic."
                    )

            is_mab_log = all(col in df.columns for col in MAB_LOG_DISTINCTIVE_COLUMNS)
            is_grid_log = any(col in df.columns for col in GRID_LOG_DISTINCTIVE_COLUMNS)

            log_type_key = None
            if is_mab_log and not is_grid_log:
                log_type_key = "mab"
            elif is_grid_log:
                log_type_key = "grid"

            if log_type_key and log_type_key not in dfs:
                dfs[log_type_key] = df
                log_filepaths[log_type_key] = log_file_path
            elif log_type_key in dfs:
                print(
                    f"Warning: Multiple logs of type '{log_type_key}' provided. Using the first one found: {log_filepaths[log_type_key]}"
                )
            else:
                print(
                    f"Warning: Could not definitively determine log type for {log_file_path}."
                )

        except pd.errors.EmptyDataError:
            print(f"Error: Log file {log_file_path} is empty.")
        except Exception as e:
            print(
                f"An error occurred during loading or type detection for {log_file_path}: {e}"
            )
            import traceback

            traceback.print_exc()

    if not dfs:
        print("No valid log files were loaded or identified. Exiting.")
        return

    plot_dir = args.plot_dir
    if plot_dir is None:
        first_valid_log_path = next(iter(log_filepaths.values()), "generic_log.csv")
        first_log_file_name = os.path.basename(first_valid_log_path)
        first_log_file_base, _ = os.path.splitext(first_log_file_name)
        plot_dir = os.path.join(
            os.path.dirname(first_valid_log_path) or ".", f"{first_log_file_base}_plots"
        )
    ensure_plot_dir(plot_dir)

    mab_summary_stats, grid_summary_stats = None, None
    mab_overall_totals_data, grid_overall_totals_data = None, None
    mab_df_full_for_comp, grid_df_full_for_comp = None, None

    current_provider = provider_model_info["provider"]
    current_model = provider_model_info["model"]

    if "mab" in dfs:
        mab_df_full_for_comp = dfs["mab"]
        mab_summary_stats, mab_overall_totals_data = analyze_mab_log(
            dfs["mab"].copy(),
            log_filepaths["mab"],
            plot_dir,
            current_provider,
            current_model,
        )
    if "grid" in dfs:
        grid_df_full_for_comp = dfs["grid"]
        grid_summary_stats, grid_overall_totals_data = analyze_grid_log(
            dfs["grid"].copy(),
            log_filepaths["grid"],
            plot_dir,
            current_provider,
            current_model,
        )

    if mab_summary_stats is not None and grid_summary_stats is not None:
        analyze_comparison(
            mab_summary_stats,
            grid_summary_stats,
            mab_overall_totals_data,
            grid_overall_totals_data,
            plot_dir,
            mab_df_full_for_comp,
            grid_df_full_for_comp,
            current_provider,
            current_model,
        )


if __name__ == "__main__":
    main()
