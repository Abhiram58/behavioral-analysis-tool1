# Combined Streamlit App: ABA/ATD Analysis with Bayesian, Bootstrapping, Mixed Model + Add-on Tab

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import (
    ttest_rel, wilcoxon, friedmanchisquare, ttest_ind, mannwhitneyu,
    shapiro, beta as beta_dist, bootstrap
)
from statsmodels.regression.mixed_linear_model import MixedLM

st.set_page_config(page_title="SCD Analysis Suite", layout="wide")
st.title("🧠 AI-Powered SCD Analysis Tool")

# Upload Instructions
st.markdown("""
### 📂 How to Upload Your Dataset
1. Prepare your dataset in **Microsoft Excel (.xlsx)** format.
2. Ensure each phase or condition is in its **own column** (e.g., `Baseline`, `Intervention`, etc.).
3. Make sure each row represents a **measurement point**.
4. If you're analyzing subject-level data, include a `Subject` column.
5. Use the sidebar on the left to **upload your Excel file**.
6. After uploading, select the sheet and configure settings as needed.
""")

uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
tabs = st.tabs(["Main Analysis", "Bayesian Logistic Add-on"])

# Helper Functions
def check_normality(data):
    stat, p_value = shapiro(data)
    return {"statistic": stat, "p_value": p_value, "is_normal": p_value > 0.05}

def visualize_normality(data, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(data, bins=10, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f"Histogram of {title}")
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot of {title}")
    sns.boxplot(data=data, ax=axes[2], color='lightblue')
    axes[2].set_title(f"Box Plot of {title}")
    plt.tight_layout()
    return fig

def display_results(results):
    st.write("### Raw Results")
    st.code(results["raw"])
    st.write("### APA-Style Results")
    st.write(results["apa"])
    st.write("### Interpretation and Feedback")
    st.write(results["feedback"])
    st.write("### Intervention Effect")
    st.write(results["effect"])
    if "pnd" in results:
        st.write(f"### Percent Non-Overlap (PND): {results['pnd']:.1f}%")

def compute_pnd(baseline, intervention):
    max_baseline = np.max(baseline)
    non_overlapping = np.sum(intervention > max_baseline)
    return (non_overlapping / len(intervention)) * 100

def descriptive_stats(df):
    stats = df.agg(['mean', 'median', 'std', 'min', 'max']).T
    stats.columns = ['Mean', 'Median', 'SD', 'Min', 'Max']
    return stats

def bayesian_logistic_regression(baseline, intervention):
    baseline = baseline.reset_index(drop=True)
    intervention = intervention.reset_index(drop=True)
    min_len = min(len(baseline), len(intervention))
    baseline = baseline.iloc[:min_len]
    intervention = intervention.iloc[:min_len]
    success = (intervention > baseline).sum()
    posterior = beta_dist(1 + success, 1 + min_len - success)
    x = np.linspace(0, 1, 1000)
    y = posterior.pdf(x)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, color='blue', alpha=0.3)
    ax.set_title("Bayesian Posterior Distribution")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    return posterior.mean(), fig

def bootstrap_ci_with_plot(data):
    result = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=1000, random_state=0)
    fig, ax = plt.subplots()
    ax.hist(result.bootstrap_distribution, bins=30, alpha=0.7, color='purple')
    ax.axvline(result.confidence_interval.low, color='red', linestyle='--')
    ax.axvline(result.confidence_interval.high, color='green', linestyle='--')
    ax.set_title("Bootstrap Distribution of Effect Size")
    ax.set_xlabel("Mean Value")
    ax.set_ylabel("Frequency")
    return result.confidence_interval, fig

def run_mixed_model(data):
    if {'Phase', 'Value', 'Subject'}.issubset(data.columns):
        data = data.copy()
        data["Label"] = data["Phase"].apply(lambda x: 0 if str(x).lower() == "baseline" else 1)
        model = MixedLM.from_formula("Value ~ Label", groups="Subject", data=data).fit()
        return model
    return None

# --- Main Tab ---
with tabs[0]:
    if uploaded_file:
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        selected_sheet = st.sidebar.selectbox("Select the sheet", sheet_names)
        data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        st.write("### Data Preview")
        st.dataframe(data.head())

        analysis_type = st.sidebar.radio("Select Analysis Type", ["ABA Reversal", "Alternating Treatment"])

        if analysis_type == "ABA Reversal":
            phases = st.sidebar.multiselect("Select Phases", data.columns.tolist(), default=data.columns.tolist())
            if len(phases) >= 2:
                phase_data = [data[phase].dropna() for phase in phases]
                min_len = min(len(p) for p in phase_data)
                trimmed_data = [p[:min_len] for p in phase_data]
                normality = [check_normality(p) for p in trimmed_data]

                st.write("### Normality Checks")
                for name, norm, values in zip(phases, normality, trimmed_data):
                    st.pyplot(visualize_normality(values, name))
                    st.write(f"{name} - Shapiro-Wilk Test: W = {norm['statistic']:.2f}, p = {norm['p_value']:.3f}")

                if len(phases) == 2:
                    all_normal = all(n["is_normal"] for n in normality)
                    valid_methods = ["Paired t-test", "Wilcoxon Signed-Rank Test", "Bayesian Analysis"] if not all_normal else ["Paired t-test", "Bayesian Analysis"]
                else:
                    valid_methods = ["Friedman Test"]

                method = st.sidebar.selectbox("Select Statistical Method", valid_methods)
                baseline, intervention = trimmed_data[0], trimmed_data[1]

                if method == "Bayesian Analysis":
                    posterior_mean, fig = bayesian_logistic_regression(baseline, intervention)
                    st.markdown("### Bayesian Logistic Regression")
                    st.pyplot(fig)
                    results = {
                        "raw": f"Posterior Mean: {posterior_mean:.3f}",
                        "apa": f"Bayesian Estimate = {posterior_mean:.3f}",
                        "feedback": "Estimates probability of intervention success.",
                        "effect": f"Bayesian analysis suggests {posterior_mean:.2%} probability of effectiveness.",
                        "pnd": compute_pnd(baseline, intervention)
                    }
                    display_results(results)

                elif method == "Paired t-test":
                    stat, p_value = ttest_rel(baseline, intervention)
                    df = len(baseline) - 1
                    results = {
                        "raw": f"Statistic: {stat}, P-value: {p_value}",
                        "apa": f"t({df}) = {stat:.2f}, p = {p_value:.3f}",
                        "feedback": "Paired t-test compares two related samples.",
                        "effect": "Significant effect." if p_value < 0.05 else "No significant effect.",
                        "pnd": compute_pnd(baseline, intervention)
                    }
                    display_results(results)

                elif method == "Wilcoxon Signed-Rank Test":
                    stat, p_value = wilcoxon(baseline, intervention)
                    results = {
                        "raw": f"Statistic: {stat}, P-value: {p_value}",
                        "apa": f"Wilcoxon W = {stat:.2f}, p = {p_value:.3f}",
                        "feedback": "Non-parametric test for two related samples.",
                        "effect": "Significant effect." if p_value < 0.05 else "No significant effect.",
                        "pnd": compute_pnd(baseline, intervention)
                    }
                    display_results(results)

                elif method == "Friedman Test":
                    stat, p_value = friedmanchisquare(*trimmed_data)
                    results = {
                        "raw": f"Statistic: {stat}, P-value: {p_value}",
                        "apa": f"Friedman χ² = {stat:.2f}, p = {p_value:.3f}",
                        "feedback": "Compares more than two related phases.",
                        "effect": "Significant differences." if p_value < 0.05 else "No significant difference."
                    }
                    display_results(results)

                st.write("### Descriptive Statistics")
                st.dataframe(descriptive_stats(data[phases].dropna()))

                st.markdown("### Bootstrap Analysis")
                ci, fig = bootstrap_ci_with_plot(data[phases[1]].dropna())
                st.pyplot(fig)
                st.write(f"**95% CI for Mean Value**: ({ci.low:.3f}, {ci.high:.3f})")

                model = run_mixed_model(data)
                if model:
                    st.markdown("### Mixed-Effects Model")
                    coef_df = pd.DataFrame({"Estimate": model.params.round(3), "P-value": model.pvalues.round(3)})
                    st.dataframe(coef_df)

        elif analysis_type == "Alternating Treatment":
            condition_col = st.sidebar.selectbox("Select Condition Column", data.columns)
            value_col = st.sidebar.selectbox("Select Value Column", [col for col in data.columns if col != condition_col])
            at_data = data[[condition_col, value_col]].dropna()
            groups = at_data[condition_col].unique()
            group1 = at_data[at_data[condition_col] == groups[0]][value_col]
            group2 = at_data[at_data[condition_col] == groups[1]][value_col]

            norm1 = check_normality(group1)
            norm2 = check_normality(group2)

            st.pyplot(visualize_normality(group1, str(groups[0])))
            st.pyplot(visualize_normality(group2, str(groups[1])))

            if norm1["is_normal"] and norm2["is_normal"]:
                at_methods = ["Independent t-test", "Bayesian Analysis"]
            else:
                at_methods = ["Mann-Whitney U Test", "Bayesian Analysis"]

            at_method = st.sidebar.selectbox("Select Statistical Method", at_methods)

            if at_method == "Bayesian Analysis":
                posterior_mean, fig = bayesian_logistic_regression(group1, group2)
                st.markdown("### Bayesian Logistic Regression")
                st.pyplot(fig)
                results = {
                    "raw": f"Posterior Mean: {posterior_mean:.3f}",
                    "apa": f"Bayesian Estimate = {posterior_mean:.3f}",
                    "feedback": "Estimates probability of condition B outperforming A.",
                    "effect": f"Bayesian analysis suggests {posterior_mean:.2%} probability of effectiveness.",
                    "pnd": compute_pnd(group1, group2)
                }
                display_results(results)

            elif at_method == "Independent t-test":
                stat, p_value = ttest_ind(group1, group2)
                results = {
                    "raw": f"Statistic: {stat}, P-value: {p_value}",
                    "apa": f"t = {stat:.2f}, p = {p_value:.3f}",
                    "feedback": "Compares two independent samples.",
                    "effect": "Significant difference." if p_value < 0.05 else "No significant difference.",
                    "pnd": compute_pnd(group1, group2)
                }
                display_results(results)

            elif at_method == "Mann-Whitney U Test":
                stat, p_value = mannwhitneyu(group1, group2)
                results = {
                    "raw": f"Statistic: {stat}, P-value: {p_value}",
                    "apa": f"U = {stat:.2f}, p = {p_value:.3f}",
                    "feedback": "Non-parametric test for two independent samples.",
                    "effect": "Significant difference." if p_value < 0.05 else "No significant difference.",
                    "pnd": compute_pnd(group1, group2)
                }
                display_results(results)

            st.write("### Descriptive Statistics")
            st.dataframe(descriptive_stats(at_data.pivot(columns=condition_col, values=value_col)))

# --- Bayesian Add-on Tab ---
with tabs[1]:
    st.header("🔍 Add-on: Bayesian Logistic Regression")
    if uploaded_file:
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        selected_sheet = st.selectbox("Select sheet for Bayesian Add-on", sheet_names)
        data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

        if len(data.columns) >= 2:
            col1 = st.selectbox("Select Baseline Column", data.columns)
            col2 = st.selectbox("Select Comparison Column", [col for col in data.columns if col != col1])

            values1 = data[col1].dropna().reset_index(drop=True)
            values2 = data[col2].dropna().reset_index(drop=True)

            if len(values1) > 1 and len(values2) > 1:
                posterior_mean, fig = bayesian_logistic_regression(values1, values2)
                st.pyplot(fig)
                st.write(f"**Posterior Mean Probability that {col2} > {col1}**: {posterior_mean:.3f}")
                st.markdown("**📝 Interpretation:** A higher posterior mean (closer to 1) suggests strong evidence that the second column outperforms the first.")
            else:
                st.warning("Not enough data in one of the columns.")
        else:
            st.warning("Please upload a sheet with at least 2 numeric columns.")
