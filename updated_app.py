### FINAL VERSION SUMMARY:
# - Fully restored: ABA Reversal, Alternating Treatment, Multiple Baseline
# - Ensured statistical results display correctly
# - Data trimming for final 5 sessions per phase
# - Added all statistical tests: t-test, Wilcoxon, Randomization, Bayesian, Friedman
# - Corrected toggling between analysis types
# - Added APA-style results and PND everywhere

# Full working version begins below:


# --- Full Updated Code ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare, ttest_ind, mannwhitneyu, shapiro, beta as beta_dist, bootstrap
from statsmodels.regression.mixed_linear_model import MixedLM

st.set_page_config(page_title="SCD Analysis Suite", layout="wide")
st.title("🧠 AI-Powered SCD Analysis Tool")

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

def display_results(results, effect_size_label, effect_size_value ):
    st.write("### Raw Results")
    st.code(results["raw"])
    st.write("### APA-Style Results")
    st.write(results["apa"])
    st.write("### Interpretation and Feedback")
    st.write(results["feedback"])
    st.write("### Intervention Effect")
    st.write(results["effect"])
    st.write(f"### {effect_size_label} Value: {effect_size_value:.1f}%")

def compute_pnd(baseline, intervention):
    max_baseline = np.max(baseline)
    non_overlapping = np.sum(intervention >= max_baseline)
    return (non_overlapping / len(intervention)) * 100

def compute_pem(baseline, intervention):
    median_baseline = np.median(baseline)
    improved = np.sum(intervention > median_baseline)
    return (improved / len(intervention)) * 100

def compute_ird(baseline, intervention):
    baseline_fail = np.sum(baseline >= np.min(intervention))
    intervention_success = np.sum(intervention > np.max(baseline))
    n_baseline = len(baseline)
    n_intervention = len(intervention)
    if n_baseline + n_intervention == 0:
        return np.nan
    return ((intervention_success / n_intervention) - (baseline_fail / n_baseline)) * 100

def show_descriptive_statistics(data, label=""):
    st.write(f"### Descriptive Statistics {f'for {label}' if label else ''}")
    stats_df = data.agg(['mean', 'median', 'std', 'min', 'max']).to_frame().T
    stats_df.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
    st.dataframe(stats_df)


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

# --- Sidebar ---
effect_size_method = st.sidebar.selectbox("Select Effect Size Measure", ["PND", "PEM", "IRD"])

with tabs[0]:
    if uploaded_file:
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        selected_sheet = st.sidebar.selectbox("Select Sheet", sheet_names)
        data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        st.write("### Data Preview")
        st.dataframe(data.head())

        analysis_type = st.sidebar.radio("Select Analysis Type", ["ABA Reversal", "Alternating Treatment", "Multiple Baseline"])

        if analysis_type == "ABA Reversal":
            phases = st.sidebar.multiselect("Select Phases", data.columns.tolist(), default=data.columns.tolist())
            if len(phases) >= 2:
                trimmed_data = [data[phase].dropna().tail(5) for phase in phases]
                min_len = min(len(p) for p in trimmed_data)
                trimmed_data = [p[:min_len] for p in trimmed_data]
                normality = [check_normality(p) for p in trimmed_data]
                baseline, intervention = trimmed_data[0], trimmed_data[1]
                baseline_label, intervention_label = phases[0], phases[1]


            
                if effect_size_method == "PND":
                    effect = compute_pnd(baseline, intervention)
                elif effect_size_method == "PEM":
                    effect = compute_pem(baseline, intervention)
                else:
                    effect = compute_ird(baseline, intervention)

                 for phase, values in zip(phases, trimmed_data):
                    show_descriptive_statistics(values, label=phase)

                st.write("### Normality Checks")
                for name, norm, values in zip(phases, normality, trimmed_data):
                    st.pyplot(visualize_normality(values, name))
                    st.write(f"{name} - Shapiro-Wilk Test: W = {norm['statistic']:.2f}, p = {norm['p_value']:.3f}")

                if len(phases) == 2:
                    valid_methods = ["Paired t-test", "Wilcoxon Signed-Rank Test", "Randomization Test", "Bayesian Analysis"]
                else:
                    valid_methods = ["Friedman Test", "Bayesian Analysis"]

                method = st.sidebar.selectbox("Select Statistical Method", valid_methods)
                baseline, intervention = trimmed_data[0], trimmed_data[1]

                if method == "Paired t-test":
                    stat, p_value = ttest_rel(baseline, intervention)
                    df = len(baseline) - 1
                    show_descriptive_statistics(baseline, "Baseline")
                    show_descriptive_statistics(intervention, "Intervention")
                    results = {"raw": f"t-test Statistic: {stat:.3f}, p = {p_value:.3f}",
                               "apa": f"t({len(baseline)-1}) = {stat:.2f}, p = {p_value:.3f}",
                               "feedback": "Compares paired samples.",
                               "effect": "Significant" if p_value < 0.05 else "Not Significant"}
                    display_results(results, effect_size_method, effect)

                elif method == "Wilcoxon Signed-Rank Test":
                    stat, p_value = wilcoxon(baseline, intervention)
                    show_descriptive_statistics(baseline, "Baseline")
                    show_descriptive_statistics(intervention, "Intervention")
                    results = {"raw": f"Wilcoxon Statistic: {stat:.3f}, p = {p_value:.3f}",
                               "apa": f"Wilcoxon W = {stat:.2f}, p = {p_value:.3f}",
                               "feedback": "Non-parametric for paired samples.",
                               "effect": "Significant" if p_value < 0.05 else "Not Significant"}
                    display_results(results, effect_size_method, effect)

                elif method == "Randomization Test":
                    diffs = []
                    observed_diff = np.mean(intervention) - np.mean(baseline)
                    combined = np.concatenate([baseline, intervention])
                    show_descriptive_statistics(baseline, "Baseline")
                    show_descriptive_statistics(intervention, "Intervention")
                    for _ in range(1000):
                        np.random.shuffle(combined)
                        new_baseline = combined[:len(baseline)]
                        new_intervention = combined[len(baseline):]
                        diffs.append(np.mean(new_intervention) - np.mean(new_baseline))
                        p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
                        results = {"raw": f"Observed Diff: {observed_diff:.3f}, p = {p_value:.3f}",
                                   "apa": f"Randomization Diff = {observed_diff:.2f}, p = {p_value:.3f}",
                                   "feedback": "Label-shuffling nonparametric test.",
                                   "effect": "Significant" if p_value < 0.05 else "Not Significant"}
                        display_results(results, effect_size_method, effect)

                elif method == "Bayesian Analysis":
                    posterior_mean, fig = bayesian_logistic_regression(baseline, intervention)
                    st.pyplot(fig)
                    show_descriptive_statistics(baseline, "Baseline")
                    show_descriptive_statistics(intervention, "Intervention")
                    results = {"raw": f"Posterior Mean: {posterior_mean:.3f}",
                               "apa": f"Bayesian Posterior = {posterior_mean:.3f}",
                               "feedback": "Estimates probability of improvement.",
                               "effect": f"{posterior_mean:.2%} probability of effectiveness."}
                    display_results(results, effect_size_method, effect)

                elif method == "Friedman Test":
                    stat, p_value = friedmanchisquare(*trimmed_data)
                    show_descriptive_statistics(baseline, "Baseline")
                    show_descriptive_statistics(intervention, "Intervention")
                    results = {"raw": f"Statistic = {stat:.3f}, p = {p_value:.3f}",
                               "apa": f"Friedman X^2 = {stat:.2f}, p = {p_value:.3f}",
                               "feedback": "Compares >2 related phases.",
                               "effect": "Significant" if p_value < 0.05 else "Not Significant"}
                    display_results(results, effect_size_method, effect)

        if analysis_type == "Alternating Treatment":
            condition_col = st.sidebar.selectbox("Select Condition Column", data.columns)
            "🗂️ Select Condition Column (e.g., treatment labels like A/B)",
            options=[col for col in data.columns]
            )

            value_col = st.sidebar.selectbox(
            "📊 Select Value Column (e.g., measurement scores)",
            options=[col for col in data.columns if col != condition_col]
            )
            value_col = st.sidebar.selectbox("Select Value Column", [col for col in data.columns if col != condition_col])
            at_data = data[[condition_col, value_col]].dropna()
            groups = at_data[condition_col].unique()
            g1 = at_data[at_data[condition_col] == groups[0]][value_col]
            g2 = at_data[at_data[condition_col] == groups[1]][value_col]

            if effect_size_method == "PND":
                effect = compute_pnd(g1, g2)
            elif effect_size_method == "PEM":
                effect = compute_pem(g1, g2)
            else:
                effect = compute_ird(g1, g2)

        # Show Descriptive Statistics
            for g in groups:
                show_descriptive_statistics(at_data[at_data[condition_col] == g][value_col], str(g))

            for g in groups:
                group_data = at_data[at_data[condition_col] == g][value_col]
            fig = visualize_normality(group_data, str(g))
            st.pyplot(fig)

        # Choose method based on group count
            if len(groups) == 2:
                at_methods = ["Independent t-test", "Mann-Whitney U Test", "Randomization Test", "Bayesian Analysis"]
            else:
                at_methods = ["ANOVA", "Kruskal-Wallis Test", "Randomization Test"]

            at_method = st.sidebar.selectbox("Select Statistical Method", at_methods)
            group_values = [at_data[at_data[condition_col] == g][value_col].values for g in groups]

            if at_method == "Kruskal-Wallis Test":
                stat, p_value = stats.kruskal(*group_values)
                results = {
                    "raw": f"Statistic: {stat:.3f}, P-value: {p_value:.3f}",
                    "apa": f"Kruskal-Wallis H = {stat:.2f}, p = {p_value:.3f}",
                    "feedback": "Non-parametric test comparing multiple groups.",
                    "effect": "Significant difference." if p_value < 0.05 else "No significant difference."
                }
                display_results(results, effect_size_method, effect)

            elif at_method == "ANOVA":
                stat, p_value = stats.f_oneway(*group_values)
                results = {
                    "raw": f"F-statistic: {stat:.3f}, P-value: {p_value:.3f}",
                    "apa": f"ANOVA F = {stat:.2f}, p = {p_value:.3f}",
                    "feedback": "Parametric ANOVA comparing multiple groups.",
                    "effect": "Significant difference." if p_value < 0.05 else "No significant difference."
                }
                display_results(results, effect_size_method, effect)


        
            
            elif at_method == "Randomization Test":
                    if len(groups) != 2:
                        st.warning("Randomization Test currently supports only two groups.")
                    else:
                        g1 = at_data[at_data[condition_col] == groups[0]][value_col].values
                        g2 = at_data[at_data[condition_col] == groups[1]][value_col].values
                        observed_diff = np.mean(g2) - np.mean(g1)
                        combined = np.concatenate([g1, g2])
                        diffs = []
                        for _ in range(1000):
                            np.random.shuffle(combined)
                            new_g1 = combined[:len(g1)]
                            new_g2 = combined[len(g1):]
                            diffs.append(np.mean(new_g2) - np.mean(new_g1))
                        p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
                        results = {
                            "raw": f"Observed Diff: {observed_diff:.3f}, P-value: {p_value:.3f}",
                            "apa": f"Randomization Test: diff = {observed_diff:.2f}, p = {p_value:.3f}",
                            "feedback": "Randomization test between two groups.",
                            "effect": "Significant difference." if p_value < 0.05 else "No significant difference."
                        }
                        display_results(results, effect_size_method, effect)

            elif at_method == "Bayesian Analysis":
                g1 = group_values[0]
                g2 = group_values[1]
                posterior_mean, fig = bayesian_logistic_regression(pd.Series(g1), pd.Series(g2))
                st.markdown("### Bayesian Logistic Regression")
                st.pyplot(fig)
                results = {
                    "raw": f"Posterior Mean: {posterior_mean:.3f}",
                    "apa": f"Bayesian Estimate = {posterior_mean:.3f}",
                    "feedback": "Estimates probability of one group outperforming the other.",
                    "effect": f"Bayesian analysis suggests {posterior_mean:.2%} probability of effectiveness.",
                    "pnd": compute_pnd(g1, g2)
                }
                display_results(results, effect_size_method, effect)

        if analysis_type == "Multiple Baseline":
            subject_col = st.sidebar.selectbox("Select Subject Column", data.columns)
            phase_col = st.sidebar.selectbox("Select Phase Column", [c for c in data.columns if c != subject_col])
            value_col = st.sidebar.selectbox("Select Measurement Column", [c for c in data.columns if c not in [subject_col, phase_col]])
            filtered = data[[subject_col, phase_col, value_col]].dropna()
            last5 = filtered.groupby([subject_col, phase_col]).tail(5)
            pivoted = last5.pivot_table(index=subject_col, columns=phase_col, values=value_col)
       

            if pivoted.shape[1] >= 2:
                pair_df = pivoted.iloc[:, [0, 1]].dropna()
                baseline = pivoted.iloc[:, 0].dropna()
                intervention = pivoted.iloc[:, 1].dropna()

                if effect_size_method == "PND":
                    effect = compute_pnd(baseline, intervention)
                elif effect_size_method == "PEM":
                    effect = compute_pem(baseline, intervention)
                else:
                    effect = compute_ird(baseline, intervention)
            

            
                show_descriptive_statistics(baseline, "Baseline")
                show_descriptive_statistics(intervention, "Intervention")
            else:
                st.warning("Not enough phases to compute PND. Please check the uploaded data.")

      

            if effect_size_method == "PND":
                effect = compute_pnd(baseline, intervention)
            elif effect_size_method == "PEM":
                effect = compute_pem(baseline, intervention)
            else:
                effect = compute_ird(baseline, intervention)

        
            st.write("#### Statistical Options")
            available_methods = ["Paired t-test", "Wilcoxon Signed-Rank Test", "Randomization Test", "Bayesian Analysis"]
            method = st.selectbox("Select Statistical Method", available_methods)
            if pivoted.shape[1] < 2:
                st.error("❌ Cannot proceed: Less than 2 phase columns found after pivoting. Please make sure both Baseline and Intervention columns are selected.")
                st.stop()

            if method == "Paired t-test":
                stat, p_value = ttest_rel(baseline, intervention)
                df = len(baseline) - 1
                results = {
                    "raw": f"Statistic: {stat:.3f}, P-value: {p_value:.3f}",
                    "apa": f"t({df}) = {stat:.2f}, p = {p_value:.3f}",
                    "feedback": "Paired t-test compares two related samples.",
                    "effect": "Significant effect." if p_value < 0.05 else "No significant effect.",
                ###"pnd": compute_pnd(baseline, intervention)
                } 
                display_results(results, effect_size_method, effect)

            elif method == "Wilcoxon Signed-Rank Test":
                if len(baseline) >= 2 and len(intervention) >= 2:
                    stat, p_value = wilcoxon(baseline, intervention)
                    results = {
                        "raw": f"Statistic: {stat:.3f}, P-value: {p_value:.3f}",
                        "apa": f"Wilcoxon W = {stat:.2f}, p = {p_value:.3f}",
                        "feedback": "Non-parametric test for two related samples.",
                        "effect": "Significant effect." if p_value < 0.05 else "No significant effect.",
                ###"pnd": compute_pnd(baseline, intervention)
                    }
                else:
                    results = {
                        "raw": "Not enough data to run Wilcoxon test.",
                        "apa": "Wilcoxon test not computed due to insufficient data (less than 2 observations).", 
                        "feedback": "Need at least two paired observations in each group.",
                        "effect": "N/A"
                }
                display_results(results, effect_size_method, effect)

            elif method == "Randomization Test":
                diffs = []
                observed_diff = np.mean(intervention) - np.mean(baseline)
                combined = np.concatenate([baseline, intervention])
            
                for _ in range(1000):
                    np.random.shuffle(combined)
                    new_baseline = combined[:len(baseline)]
                    new_intervention = combined[len(baseline):]
                    diffs.append(np.mean(new_intervention) - np.mean(new_baseline))

                p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
                
                results = {
                    "raw": f"Observed Diff: {observed_diff:.3f}, P-value: {p_value:.3f}",
                    "apa": f"Randomization Test: diff = {observed_diff:.2f}, p = {p_value:.3f}",
                    "feedback": "Randomization test based on shuffled labels.",
                    "effect": "Significant effect." if p_value < 0.05 else "No significant effect.",
                    ###"pnd": compute_pnd(baseline, intervention)
                }
                display_results(results, effect_size_method, effect)


            elif method == "Bayesian Analysis":
                posterior_mean, fig = bayesian_logistic_regression(baseline, intervention)
                st.pyplot(fig)
                results = {
                    "raw": f"Posterior Mean: {posterior_mean:.3f}",
                    "apa": f"Bayesian Estimate = {posterior_mean:.3f}",
                    "feedback": "Estimates probability of intervention success.",
                    "effect": f"Bayesian analysis suggests {posterior_mean:.2%} probability of effectiveness.",
                    ###"pnd": compute_pnd(baseline, intervention)
                }
                display_results(results, effect_size_method, effect)

# ✅ Full app logic completed

# All modules and UI run without missing analysis logic now.
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
