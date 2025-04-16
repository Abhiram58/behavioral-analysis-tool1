# Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_rel, wilcoxon, shapiro, friedmanchisquare, ttest_ind, mannwhitneyu
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import beta as beta_dist

# Streamlit page configuration
st.set_page_config(page_title="Statistical Analysis Tool", layout="wide")

# Title and instructions
st.title("AI-Powered Statistical Analysis Tool")
st.subheader("Instructions for File Upload")
st.write("1. Ensure your dataset is in **Excel format (.xlsx)**.")
st.write("2. Include appropriate column names (e.g., 'Baseline', 'Intervention', etc.).")
st.write("3. If unsure about naming conventions, consult the data preview after uploading.")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

# Function to check normality using Shapiro-Wilk Test
def check_normality(data):
    stat, p_value = shapiro(data)
    return {"statistic": stat, "p_value": p_value, "is_normal": p_value > 0.05}

# Function to visualize normality using histogram, QQ plot, and box plot
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

# Function to display results in Raw, APA-style, Interpretation, Effect, and optionally PND
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

# Function to calculate Percent Non-Overlap (PND) between baseline and intervention
def compute_pnd(baseline, intervention):
    max_baseline = np.max(baseline)
    non_overlapping = np.sum(intervention > max_baseline)
    return (non_overlapping / len(intervention)) * 100

# Function to calculate descriptive statistics for each column
def descriptive_stats(df):
    stats = df.agg(['mean', 'median', 'std', 'min', 'max']).T
    stats.columns = ['Mean', 'Median', 'SD', 'Min', 'Max']
    return stats

# Main execution if file is uploaded
if uploaded_file:
     # Show available sheets and let user select one
    sheet_names = pd.ExcelFile(uploaded_file).sheet_names
    selected_sheet = st.sidebar.selectbox("Select the sheet", sheet_names)
    data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    
     # Display data preview
    st.write("### Data Preview")
    st.write(data.head())

    # Select design type
    analysis_type = st.sidebar.radio("Select Analysis Type", ["ABA Reversal", "Alternating Treatment"])

    # ABA Reversal Design Analysis
    if analysis_type == "ABA Reversal":
         # Select phases (e.g., Baseline, Intervention)
        phases = st.sidebar.multiselect("Select Phases for Analysis", data.columns.tolist(), default=data.columns.tolist())

        if len(phases) >= 2:
             # Trim all phase data to same length          
            phase_data = [data[phase].dropna() for phase in phases]
            min_len = min(len(p) for p in phase_data)
            trimmed_data = [p[:min_len] for p in phase_data]
            normality = [check_normality(p) for p in trimmed_data]
           
              # Display normality visualizations and results
            st.write("### Normality Checks")
            for name, norm, values in zip(phases, normality, trimmed_data):
                st.pyplot(visualize_normality(values, name))
                st.write(f"{name} - Shapiro-Wilk Test: W = {norm['statistic']:.2f}, p = {norm['p_value']:.3f}, Normality: {'Yes' if norm['is_normal'] else 'No'}")

            # Choose valid methods based on normality and number of phases
            if len(phases) == 2:
                all_normal = all(n["is_normal"] for n in normality)
                if all_normal:
                    valid_methods = ["Paired t-test", "Bayesian Analysis"]
                else:
                    valid_methods = ["Wilcoxon Signed-Rank Test", "Bayesian Analysis"]
            else:
                valid_methods = ["Friedman Test"]

            method = st.sidebar.selectbox("Select Statistical Method", valid_methods)

             # Apply appropriate test and display results
            if method == "Paired t-test":
                stat, p_value = ttest_rel(trimmed_data[0], trimmed_data[1])
                df = len(trimmed_data[0]) - 1
                results = {
                    "raw": f"Statistic: {stat}, P-value: {p_value}",
                    "apa": f"t({df}) = {stat:.2f}, p = {p_value:.3f}",
                    "feedback": "A paired t-test compares means between phases assuming normality.",
                    "effect": "The intervention had a statistically significant effect." if p_value < 0.05 else "No significant effect.",
                    "pnd": compute_pnd(trimmed_data[0], trimmed_data[1])
                }
                display_results(results)

            elif method == "Wilcoxon Signed-Rank Test":
                stat, p_value = wilcoxon(trimmed_data[0], trimmed_data[1])
                results = {
                    "raw": f"Statistic: {stat}, P-value: {p_value}",
                    "apa": f"Wilcoxon W = {stat:.2f}, p = {p_value:.3f}",
                    "feedback": "Wilcoxon test compares matched phases without assuming normality.",
                    "effect": "Statistically significant effect." if p_value < 0.05 else "No significant effect.",
                    "pnd": compute_pnd(trimmed_data[0], trimmed_data[1])
                }
                display_results(results)

            elif method == "Bayesian Analysis":
                prior_alpha, prior_beta = 1, 1
                success = (trimmed_data[1] > trimmed_data[0]).sum()
                trials = len(trimmed_data[1])
                posterior_alpha = prior_alpha + success
                posterior_beta = prior_beta + trials - success
                posterior_mean = beta_dist(posterior_alpha, posterior_beta).mean()
                posterior_prob = f"{posterior_mean:.2%}"
                results = {
                    "raw": f"Posterior Mean: {posterior_mean:.3f}",
                    "apa": f"Bayesian Estimate = {posterior_mean:.3f}",
                    "feedback": "Estimates probability of intervention success.",
                    "effect": f"Bayesian analysis suggests {posterior_prob} probability of effectiveness.",
                    "pnd": compute_pnd(trimmed_data[0], trimmed_data[1])
                }
                display_results(results)

            elif method == "Friedman Test":
                stat, p_value = friedmanchisquare(*trimmed_data)
                df = len(trimmed_data[0]) - 1
                pnd_value = compute_pnd(trimmed_data[0], trimmed_data[1]) if len(trimmed_data) >= 2 else None
                results = {
                    "raw": f"Statistic: {stat}, P-value: {p_value}",
                    "apa": f"Friedman χ² = {stat:.2f}, p = {p_value:.3f}",
                    "feedback": "Friedman test compares >2 related phases.",
                    "effect": "Statistically significant differences." if p_value < 0.05 else "No significant difference.",
                    "pnd": pnd_value
                }
                display_results(results)

             # Show descriptive stats for selected phases
            st.write("### Descriptive Statistics")
            desc_data = data[phases].dropna()
            st.dataframe(descriptive_stats(desc_data))

     # Alternating Treatment Design Analysis
    elif analysis_type == "Alternating Treatment":
        # Select columns representing condition label and value
        condition_col = st.sidebar.selectbox("Select Condition Column", data.columns)
        value_col = st.sidebar.selectbox("Select Value Column", [col for col in data.columns if col != condition_col])

        # Drop missing values and separate into groups
        at_data = data[[condition_col, value_col]].dropna()
        groups = at_data[condition_col].unique()
        group1 = at_data[at_data[condition_col] == groups[0]][value_col]
        group2 = at_data[at_data[condition_col] == groups[1]][value_col]

        # Perform normality check for both groups
        norm1 = check_normality(group1)
        norm2 = check_normality(group2)

         # Display normality visualizations
        st.write("### Normality Checks")
        st.pyplot(visualize_normality(group1, str(groups[0])))
        st.write(f"{groups[0]} - Shapiro-Wilk Test: W = {norm1['statistic']:.2f}, p = {norm1['p_value']:.3f}")
        st.pyplot(visualize_normality(group2, str(groups[1])))
        st.write(f"{groups[1]} - Shapiro-Wilk Test: W = {norm2['statistic']:.2f}, p = {norm2['p_value']:.3f}")

        # Select test method based on normality
        if norm1["is_normal"] and norm2["is_normal"]:
            at_methods = ["Independent t-test", "Bayesian Analysis"]
        else:
            at_methods = ["Mann-Whitney U Test", "Bayesian Analysis"]

        at_method = st.sidebar.selectbox("Select Statistical Method", at_methods)

         # Run and display selected test
        if at_method == "Independent t-test":
            stat, p_value = ttest_ind(group1, group2)
            results = {
                "raw": f"Statistic: {stat}, P-value: {p_value}",
                "apa": f"t = {stat:.2f}, p = {p_value:.3f}",
                "feedback": "Independent t-test compares group means assuming normality.",
                "effect": "Statistically significant difference." if p_value < 0.05 else "No significant difference.",
                "pnd": compute_pnd(group1, group2)
            }
            display_results(results)

        elif at_method == "Mann-Whitney U Test":
            stat, p_value = mannwhitneyu(group1, group2)
            results = {
                "raw": f"Statistic: {stat}, P-value: {p_value}",
                "apa": f"U = {stat:.2f}, p = {p_value:.3f}",
                "feedback": "Mann-Whitney U test compares distributions without normality.",
                "effect": "Statistically significant difference." if p_value < 0.05 else "No significant difference.",
                "pnd": compute_pnd(group1, group2)
            }
            display_results(results)

        elif at_method == "Bayesian Analysis":
            prior_alpha, prior_beta = 1, 1
            success = (group2.values > group1.values[:len(group2)]).sum()
            trials = len(group2)
            posterior_alpha = prior_alpha + success
            posterior_beta = prior_beta + trials - success
            posterior_mean = beta_dist(posterior_alpha, posterior_beta).mean()
            posterior_prob = f"{posterior_mean:.2%}"
            results = {
                "raw": f"Posterior Mean: {posterior_mean:.3f}",
                "apa": f"Bayesian Estimate = {posterior_mean:.3f}",
                "feedback": "Bayesian analysis estimates probability of condition B outperforming A.",
                "effect": f"Bayesian analysis suggests {posterior_prob} probability of effectiveness.",
                "pnd": compute_pnd(group1, group2)
            }
            display_results(results)

        # Show descriptive statistics for both groups
        st.write("### Descriptive Statistics")
        st.dataframe(descriptive_stats(at_data.pivot(columns=condition_col, values=value_col)))
# command to run this file --> streamlit run < file.name >
