import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
import tiktoken
import umap
import os
import re


TEST_CASE_DATA_FOLDER = os.path.join("..", "data", "generated_datasets")
CONDITIONAL_RESULTS_FOLDER = os.path.join("..", "data", "conditional_decision_results")
DECISION_DATA_FOLDER = os.path.join("..", "data", "decision_results")
PLOT_OUTPUT_FOLDER = os.path.join("..", "plots")
TEST_CASE_DATASET = os.path.join("..", "data", "full_dataset.csv")

BIAS_NAME_MAPPING = {
    'Escalation Of Commitment': 'Escalation of Commitment', 
    'Illusion Of Control': 'Illusion of Control',
    'Self Serving Bias': 'Self-Serving Bias',
    'In Group Bias': 'In-Group Bias',
    'Status Quo Bias': 'Status-Quo Bias'
}

MODEL_NAME_MAPPING = {
    'gpt-4o-2024-08-06': 'GPT-4o',
    'gpt-4o-mini-2024-07-18': 'GPT-4o mini',
    'gpt-3.5-turbo-0125': 'GPT-3.5 Turbo',
    'meta-llama/Meta-Llama-3.1-405B-Instruct': 'Llama 3.1 405B',
    'meta-llama/Meta-Llama-3.1-70B-Instruct': 'Llama 3.1 70B',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'Llama 3.1 8B',
    'meta-llama/Llama-3.2-3B-Instruct': 'Llama 3.2 3B',
    'meta-llama/Llama-3.2-1B-Instruct': 'Llama 3.2 1B',
    'claude-3-haiku-20240307': 'Claude 3 Haiku',
    'models/gemini-1.5-pro': 'Gemini 1.5 Pro',
    'models/gemini-1.5-flash': 'Gemini 1.5 Flash',
    'google/gemma-2-27b-it': 'Gemma 2 27B',
    'google/gemma-2-9b-it': 'Gemma 2 9B',
    'mistral-small-2409': 'Mistral Small',
    'mistral-large-2407': 'Mistral Large',
    'microsoft/WizardLM-2-8x22B': 'WizardLM-2 8x22B',
    'microsoft/WizardLM-2-7B': 'WizardLM-2 7B',
    'accounts/fireworks/models/phi-3-vision-128k-instruct': 'Phi-3.5',
    'Qwen/Qwen2.5-72B-Instruct': 'Qwen2.5 72B',
    'accounts/yi-01-ai/models/yi-large': 'Yi-Large',
    'random-model': 'Random'
}

MODEL_DEVELOPER_MAPPING = {
    'GPT-4o': 'OpenAI',
    'GPT-4o mini': 'OpenAI',
    'GPT-3.5 Turbo': 'OpenAI',
    'Llama 3.1 405B': 'Meta',
    'Llama 3.1 70B': 'Meta',
    'Llama 3.1 8B': 'Meta',
    'Llama 3.2 3B': 'Meta',
    'Llama 3.2 1B': 'Meta',
    'Claude 3 Haiku': 'Anthropic',
    'Gemini 1.5 Pro': 'Google',
    'Gemini 1.5 Flash': 'Google',
    'Gemma 2 27B': 'Google',
    'Gemma 2 9B': 'Google',
    'Mistral Small': 'Mistral',
    'Mistral Large': 'Mistral',
    'WizardLM-2 8x22B': 'Microsoft',
    'WizardLM-2 7B': 'Microsoft',
    'Phi-3.5': 'Microsoft',
    'Qwen2.5 72B': 'Alibaba',
    'Yi-Large': '01.AI',
    'Random': 'None'
}

MODEL_SIZE_MAPPING = {
    'GPT-4o': 200, # Assumption (real size not published)
    'GPT-4o mini': 10, # Assumption (real size not published)
    'GPT-3.5 Turbo': 175,
    'Llama 3.1 405B': 405,
    'Llama 3.1 70B': 70,
    'Llama 3.1 8B': 8,
    'Llama 3.2 3B': 3,
    'Llama 3.2 1B': 1,
    'Claude 3 Haiku': 20, # Assumption (real size not published)
    'Gemini 1.5 Pro': 200, # Assumption (real size not published)
    'Gemini 1.5 Flash': 30, # Assumption (real size not published)
    'Gemma 2 27B': 27,
    'Gemma 2 9B': 9,
    'Mistral Large': 123,
    'Mistral Small': 22,
    'WizardLM-2 8x22B': 176,
    'WizardLM-2 7B': 7,
    'Phi-3.5': 4.2,
    'Qwen2.5 72B': 72,
    'Yi-Large': 34,
    'Random': 0
}

# Model scores from Chatbot Arena
MODEL_SCORE_MAPPING = {
    'GPT-4o': 1264,
    'GPT-4o mini': 1273,
    'GPT-3.5 Turbo': 1106,
    'Llama 3.1 405B': 1267,
    'Llama 3.1 70B': 1248,
    'Llama 3.1 8B': 1172,
    'Llama 3.2 3B': 1102,
    'Llama 3.2 1B': 1054,
    'Claude 3 Haiku': 1179,
    'Gemini 1.5 Pro': 1304,
    'Gemini 1.5 Flash': 1265,
    'Gemma 2 27B': 1218,
    'Gemma 2 9B': 1189,
    'Mistral Small': None,
    'Mistral Large': 1251,
    'WizardLM-2 8x22B': None,
    'WizardLM-2 7B': None,
    'Phi-3.5': None,
    'Qwen2.5 72B': 1257,
    'Yi-Large': 1212,
    'Random': None
}

MODEL_ORDER = list(MODEL_NAME_MAPPING.values())


def load_decision_data(folder_path: str = DECISION_DATA_FOLDER, format: bool = True) -> pd.DataFrame:
    """
    Loads a single dataframe with all decision results.
    """

    # List to store dataframes
    dataframes = []

    # List to store the column names of each dataframe
    columns_list = []

    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):  # Only process CSV files
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)  # Load the CSV into a dataframe
            dataframes.append(df)  # Store the dataframe
            columns_list.append(set(df.columns))  # Store the columns as a set for comparison

    # Find the common columns across all dataframes
    common_columns = set.intersection(*columns_list)

    # Filter each dataframe to only keep the common columns
    filtered_dataframes = [df[list(common_columns)] for df in dataframes]

    # Concatenate the filtered dataframes into one large dataframe
    df = pd.concat(filtered_dataframes, ignore_index=True)

    # Format the decision data if requested
    if format:
        df = format_decision_data(df)

    # As of October 31, 2025, re-compute the bias scores for Conservatism (previously flip_treatment was mistakenly set to True) 
    conservatism_rows = df[df['bias'] == 'Conservatism']["model"].index
    df = recompute_bias_scores(df, conservatism_rows, k=-1, x_1=0, x_2=0, flip_control=False, flip_treatment=False)

    return df


def load_test_case_data(file_path: str = TEST_CASE_DATASET) -> pd.DataFrame:
    """
    Loads a pandas DataFrame with all test cases.
    """

    # Load the dataset with all test cases
    df_tests = pd.read_csv(file_path)
    return df_tests


def load_model_bias_data(df_decisions: pd.DataFrame = None, df_tests: pd.DataFrame = None) -> pd.DataFrame:
    """
    Loads a pandas DataFrame with bias results of all models (each bias is a separate column).
    """

    # If no dataframe with decision results is passed, load it from files
    if df_decisions is None:
        df_decisions = load_decision_data()

    # If no dataframe with test cases is passed, load it from files
    if df_tests is None:
        df_tests = load_test_case_data()

    # Step 1: Merge a scenario column onto the decision results dataframe
    df = df_decisions[["id", "bias", "individual_score", "model"]].sort_values(by=["model", "id"]).merge(df_tests[["id", "scenario"]])

    # Step 2: Sort the dataframe by "model", "bias", "scenario", and "id" to ensure the individual_scores are in order
    df = df.sort_values(by=["model", "scenario", "bias", "id"])

    # Step 3: Create a new column for each unique value of "bias" and pivot the table
    df = df.pivot_table(
        index=["model", "scenario"],  # Group by "model" and "scenario"
        columns="bias",               # Create one column for each unique value in "bias"
        values="individual_score",    # The value in each cell is the "individual_score"
        aggfunc=lambda x: list(x)     # Merge rows into a list to preserve order of multiple values
    )

    # Step 4: Split the lists into 5 individual rows
    df = df.apply(lambda row: row.explode(), axis=0).reset_index()

    return df


def load_model_failures(df_decisions: pd.DataFrame):
    """
    Loads a DataFrame capturing the success rate (1 - failure rate) of all models.
    """

    # Calculate the average success rate per model
    df_failures = df_decisions[["model", "status"]].copy()
    df_failures["status"] = df_failures["status"].map({"OK": 1, "ERROR": 0}) * 100.0
    df_failures = df_failures.groupby("model").mean()
    df_failures = df_failures.sort_values(by="status", ascending=False).reset_index()

    # Rename the column
    df_failures = df_failures.rename(columns={"status": "Valid Answers"})

    return df_failures


def load_model_answer_length(df_decisions: pd.DataFrame):
    """
    Loads a DataFrame capturing the average length (in tokens) of answers per model.
    """

    # Load tiktoken tokenizer encodings
    encoding = tiktoken.get_encoding("cl100k_base")

    # Define a reusable function for counting tokens
    def count_tokens(s: str):
        return len(encoding.encode(s))

    # Count the number of tokens in the model answer
    df_lengths = df_decisions[["model", "control_answer", "treatment_answer"]].copy()
    df_lengths["tokens_control_answer"] = df_decisions["control_answer"].astype(str).apply(count_tokens)
    df_lengths["tokens_treatment_answer"] = df_decisions["treatment_answer"].astype(str).apply(count_tokens)

    # Sum up the token counts
    df_lengths["Average Output Tokens"] = df_lengths["tokens_control_answer"] + df_lengths["tokens_treatment_answer"]
    
    # Calculate the average output token count per model
    df_lengths = df_lengths[["model", "Average Output Tokens"]].groupby("model").mean().reset_index().sort_values(by="Average Output Tokens", ascending=False)

    return df_lengths


def load_model_characteristics(df_decisions: pd.DataFrame, df_biasedness: pd.DataFrame, incl_failures: bool = False, incl_lengths: bool = False, incl_random: bool = False, fill_na_scores: bool = True):
    """
    Loads a DataFrame with a summary of model characteristics.
    """

    # Put all relevant information about the models into a single DataFrame
    df_mean_abs_bias = calculate_mean_absolute(df_biasedness, by="model", col_name="Bias")
    df_mean_abs_bias["Parameters"] = df_mean_abs_bias["model"].map(MODEL_SIZE_MAPPING)
    df_mean_abs_bias["Developer"] = df_mean_abs_bias["model"].map(MODEL_DEVELOPER_MAPPING)
    df_mean_abs_bias["Score"] = df_mean_abs_bias["model"].map(MODEL_SCORE_MAPPING)

    # If requested, load additional information on the % of failed answers per model
    if incl_failures:
        df_failures = load_model_failures(df_decisions)
        df_mean_abs_bias = df_mean_abs_bias.merge(df_failures)

    # If requested, load additional information on the average output tokens per model
    if incl_lengths:
        df_lengths = load_model_answer_length(df_decisions)
        df_mean_abs_bias = df_mean_abs_bias.merge(df_lengths)

    # Rename some columns
    df_mean_abs_bias = df_mean_abs_bias.rename(columns={"Bias": "Mean Absolute Bias", "Score": "Chatbot Arena Score"})

    # If requested, fill in missing score values with the mean score and add an asterisk to the model name
    if fill_na_scores:
        missing_scores = df_mean_abs_bias["Chatbot Arena Score"].isna()
        mean_score = df_mean_abs_bias["Chatbot Arena Score"].mean()
        df_mean_abs_bias["Chatbot Arena Score"] = df_mean_abs_bias["Chatbot Arena Score"].fillna(mean_score)
        df_mean_abs_bias.loc[missing_scores, "model"] = df_mean_abs_bias.loc[missing_scores, "model"] + '*'

    # Unless requested, exclude the Random model
    if not incl_random:
        df_mean_abs_bias = df_mean_abs_bias[df_mean_abs_bias["model"] != "Random*"].reset_index(drop=True)

    return df_mean_abs_bias


def recompute_bias_scores(df: pd.DataFrame, indices: list[int], k: int = -1, x_1: int = 0, x_2: int = 0, flip_control: bool = False, flip_treatment: bool = False) -> pd.DataFrame:
    """
    Recomputes the bias scores in a dataframe with decision results.
    """
    # Extract the control and treatment answers for the specified indices
    control_answers = df.loc[indices, 'control_decision'].to_numpy()
    treatment_answers = df.loc[indices, 'treatment_decision'].to_numpy()

    # Convert all metric parameters to numpy arrays for vectorized computation
    k = np.array([k])
    x_1 = np.array([x_1])
    x_2 = np.array([x_2])

    # Compute the metric values using the formula
    delta_control_abs, delta_treatment_abs = np.abs(control_answers - x_1), np.abs(treatment_answers - x_2)
    metric_values = k * (delta_control_abs - delta_treatment_abs) / (np.maximum(delta_control_abs, delta_treatment_abs) + 1e-8)

    # Update the bias scores for the specified indices with the computed metric values
    df = df.copy()
    df.loc[indices, 'individual_score'] = metric_values

    return df


def format_decision_data(df: pd.DataFrame, format_bias_names: bool = True, format_model_names: bool = True) -> pd.DataFrame:
    """
    Formats the bias and model names in a dataframe with decision results.
    """

    if format_bias_names:
        # Format bias names properly
        df["bias"] = df["bias"].apply(
            lambda x: re.sub(r'([a-z])([A-Z])', r'\1 \2', x)
        ).replace(BIAS_NAME_MAPPING)

    if format_model_names:
        df["model"] = df["model"].replace(MODEL_NAME_MAPPING)

    return df


def impute_missing_values(df: pd.DataFrame):
    """
    Imputes all missing values in the dataframe.
    """

    # Impute missing values with mean values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Identify columns with missing values
    nan_cols = df.columns[df.isna().any()]
    non_nan_cols = df.columns.drop(nan_cols)

    # Impute the missing values in the numeric columns
    df_imputed = pd.DataFrame(imputer.fit_transform(df[nan_cols]), columns=nan_cols)
    df_imputed[non_nan_cols] = df[non_nan_cols]
    df_imputed = df_imputed[df.columns]

    return df_imputed


def group_by(df: pd.DataFrame, by: str, agg: str = "mean"):
    """
    Groups the DataFrame by a specific column and applies an aggregation to all other numeric columns. Drops all non-numeric columns that are not used for the grouping.
    """

    # Assemble a list of all numeric columns and the column the DataFrame shall be grouped by
    keep_cols = df.select_dtypes(include=['number']).columns
    keep_cols = list(keep_cols)
    keep_cols.append("model")

    # Group the dataframe by the selected column and apply the aggregation function to all other numeric columns
    df_grouped = df[keep_cols].groupby("model").agg(agg).reset_index()

    return df_grouped


def calculate_mean_absolute(df: pd.DataFrame, by: str, col_name: str):
    """
    Calculates mean absolute values of all numeric columns by (1) grouping by the "by" column and (2) aggregating all numeric columns into a single "col_name" column.
    """

    # Assemble a list of all numeric columns and the column the DataFrame shall be grouped by
    num_cols = df.select_dtypes(include=['number']).columns
    keep_cols = list(num_cols)
    keep_cols.append("model")

    # Convert all numeric values to absolute values
    df[num_cols] = df[num_cols].abs()

    # Group the dataframe by the selected column and apply the aggregation function to all other numeric columns
    df_grouped = df[keep_cols].groupby("model").agg("mean").reset_index()

    # Aggregate all numeric columns into one by taking the mean
    df_grouped[col_name] = df_grouped[num_cols].mean(axis=1)
    df_grouped = df_grouped.drop(columns=num_cols)

    return df_grouped


def reduce_with_pca(df: pd.DataFrame, n_components: int = 2):
    """
    Performs a PCA dimensionality reduction.
    """

    # Instantiate a PCA instance
    pca = PCA(n_components=n_components)

    # Select all numeric columns
    num_cols = df.select_dtypes(include=['number']).columns

    # Run the PCA
    pca_result = pd.DataFrame(pca.fit_transform(df[num_cols]), columns=[f"PCA Component {i+1}" for i in range(n_components)])

    return pca_result


def reduce_with_umap(df: pd.DataFrame, n_components: int = 2):
    """
    Performs a UMAP dimensionality reduction.
    """

    # Instantiate a UMAP instance
    umap_reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)

    # Select all numeric columns
    num_cols = df.select_dtypes(include=['number']).columns

    # Run UMAP
    umap_result = pd.DataFrame(umap_reducer.fit_transform(df[num_cols]), columns=[f"UMAP Component {i+1}" for i in range(n_components)])

    return umap_result


def cluster_with_kmeans(df: pd.DataFrame, n_clusters: int, scale_first: bool = False):
    """
    Clusters the data with K-means.
    """

    # Instantiate a K-means instance
    kmeans = KMeans(n_clusters=4, random_state=42)

    # Select all numeric columns
    num_cols = df.select_dtypes(include=['number']).columns

    # If requested, apply standard scaling to the data first
    if scale_first:
        scaler = StandardScaler()
        df[num_cols] = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

    # Cluster the data
    clusters = kmeans.fit_predict(df[num_cols])

    return clusters


def cluster_with_hdbscan(df: pd.DataFrame, **kwargs):
    """
    Clusters the data with HDBSCAN.
    """

    # Instantiate an HDBSCAN instance
    hdbcan = HDBSCAN(**kwargs)

    # Select all numeric columns
    num_cols = df.select_dtypes(include=['number']).columns

    # Cluster the data
    clusters = hdbcan.fit_predict(df[num_cols])

    return clusters


def plot_bias_heatmap(df: pd.DataFrame, model_order: list[str] = MODEL_ORDER, abs: bool = False, add_avg_abs: bool = True, legend: bool = True, agg: str = 'mean', figsize: tuple[float] = (11, 12), save_plot: bool = True):
    """
    Plots a heatmap showing the bias scores of all model-bias combinations
    """

    # If requested, convert all scores to absolute values
    if abs:
        df = df.copy()
        df['individual_score'] = df['individual_score'].abs()

    # Pivot the data to create the matrix needed for the heatmap
    heatmap_data = df.pivot_table(values='individual_score', index='bias', columns='model', aggfunc=agg)

    # Add the 'Average' column to the right (average of each row)
    heatmap_data['Average'] = heatmap_data.mean(axis=1)

    # Sort the rows by the 'Average' column in descending order
    heatmap_data = heatmap_data.sort_values(by='Average', ascending=False)

    # Add the 'Average' row at the bottom (average of each column)
    average_row = heatmap_data.mean(axis=0)
    heatmap_data.loc['Average'] = average_row

    # If requested, perform similar calculations to add a 'Average Absolute' row at the bottom
    if add_avg_abs:
        df_abs = df[["model", "bias", "individual_score"]].copy()
        df_abs["individual_score"] = df_abs["individual_score"].abs()
        df_abs = df_abs.pivot_table(values='individual_score', index='bias', columns='model', aggfunc='mean')
        df_abs['Average'] = df_abs.mean(axis=1)
        abs_values = df_abs.mean(axis=0)
        heatmap_data.loc['Average Absolute'] = abs_values

    # Reindex the dataframe with the custom column order
    heatmap_data = heatmap_data.reindex(columns=(model_order + ['Average']))

    # Create the figure and set up subplots with appropriate spacing
    fig, ax = plt.subplots(figsize=figsize)

    # Create the heatmap without the default colorbar
    sns.heatmap(
        heatmap_data.round(2),  # Round values to 2 decimal places
        cmap=sns.diverging_palette(220, 20, as_cmap=True),  # Use a red-white-blue color palette
        center=0,  # Center the colormap around 0
        annot=True,  # Display values in cells, rounded to two decimals
        fmt=".2f",   # Format annotation to 2 decimal places
        vmin=-1.0, vmax=1.0,  # Ensure the color scale covers the range [-1.0, 1.0]
        cbar=False,  # Disable the default colorbar
        linewidths=0,  # Remove inner gridlines
        linecolor='white',  # Prevent additional grid lines from showing
        ax=ax
    )

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Rotate the column labels by 90 degrees and align them properly
    ax.xaxis.tick_top()
    plt.xticks(rotation=90, ha='center')

    # Remove black border around the entire plot
    for spine in ax.spines.values():
        spine.set_visible(False)  # Set borders to invisible

    # Remove the black lines around the white gap for 'Average' row and column
    ax.hlines([heatmap_data.shape[0] - 1], *ax.get_xlim(), color='white', linewidth=5)  # Row gap
    if add_avg_abs:
        ax.hlines([heatmap_data.shape[0] - 2], *ax.get_xlim(), color='white', linewidth=5)  # Row gap
    ax.vlines([heatmap_data.shape[1] - 1], *ax.get_ylim(), color='white', linewidth=5)  # Column gap

    # If requested, add a colorbar legend
    if legend:
        # Manually add the colorbar below the heatmap
        cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])  # Adjust [left, bottom, width, height]
        cbar = plt.colorbar(ax.collections[0], cax=cbar_ax, orientation='horizontal', label='Bias')

        # Remove the border around the colorbar
        cbar.outline.set_visible(False)

    # Adjust the space to avoid overlap using subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)  # Modify these values if needed

    # Save the plot
    if save_plot:
        plt.savefig(os.path.join(PLOT_OUTPUT_FOLDER, "bias_heatmap.pdf"), format='pdf', bbox_inches='tight')

    # Display the heatmap
    plt.show()


def plot_scatter(df: pd.DataFrame, label: str, dot_size: int = 1, dot_alpha: float = 0.5, save_plot: bool = True):
    """
    Plots a scatter plot.
    """

    # Make sure the label column is in string format
    df[label] = df[label].astype(str)

    # Select all numeric columns
    num_cols = df.select_dtypes(include=['number']).columns

    # If there aren't exactly two numeric columns, raise an error
    if len(num_cols) != 2:
        raise ValueError(f"DataFrame must contain exactly two numeric columns. Found {len(num_cols)}.")

    # Assign a color to each unique label
    unique_labels = sorted(df[label].unique())
    label_mapping = {l: idx for idx, l in enumerate(unique_labels)}
    
    # Use consistent color mapping based on label index
    colors = [plt.cm.Spectral(label_mapping[l] / len(unique_labels)) for l in df[label]]

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x=df[num_cols[0]], y=df[num_cols[1]], c=colors, s=dot_size, alpha=dot_alpha)

    # Add a legend with consistent colors
    handles = [plt.Line2D([0], [0], marker='o', color=plt.cm.Spectral(i / len(unique_labels)), markersize=10, linestyle='') for i in range(len(unique_labels))]
    plt.legend(handles, unique_labels, title=label, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set chart title and axis labels
    plt.title("Cluster Plot")
    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])

    # Save the plot
    if save_plot:
        plt.savefig(os.path.join(PLOT_OUTPUT_FOLDER, "cluster_plot.pdf"), format='pdf', bbox_inches='tight')

    # Display the heatmap
    plt.show()


def plot_dendrogram(df: pd.DataFrame, label: str, scale_first: bool = False, method: str = 'complete', metric: str = 'euclidean', n_clusters: int = 10, save_plot: bool = True):
    """
    Performs agglomerative clustering and plots the results as a dendrogram.
    """

    # Select all numeric columns for performing the clustering
    num_cols = df.select_dtypes('number').columns

    # If requested, scale the data first
    if scale_first:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Perform Agglomerative Clustering and compute the linkage matrix
    linkage_matrix = sch.linkage(df[num_cols], method=method, metric=metric)

    # Create a color palette from the 'Spectral' colormap with alpha = 0.7
    cmap = plt.get_cmap('Spectral', n_clusters)
    # colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    colors = [(*cmap(i / n_clusters)[:3], 0.7) for i in range(n_clusters)]

    # Function to apply colors to dendrogram
    def color_func(x):
        return mcolors.to_hex(colors[x % n_clusters])

    # Plot the dendrogram
    plt.figure(figsize=(5, 5))
    sch.dendrogram(linkage_matrix, labels=df[label].values, orientation='right', link_color_func=color_func)
    plt.xlabel(f"{metric.title()} Distance")
    
    # Save the plot
    if save_plot:
        plt.savefig(os.path.join(PLOT_OUTPUT_FOLDER, "dendrogram.pdf"), format='pdf', bbox_inches='tight')

    # Display the heatmap
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_plot: bool = True):
    """
    Plots a correlation matrix heatmap for all numeric columns in the DataFrame.
    """

    # Step 1: Select all numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Step 2: Calculate the correlation matrix
    correlation_matrix = df[numeric_columns].corr()

    # Step 3: Plot the correlation matrix using seaborn's heatmap
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(correlation_matrix, annot=False, fmt=".1f", cmap=sns.diverging_palette(220, 20, as_cmap=True), center=0, cbar=True)
    
    # Step 4: Set plot title and labels
    plt.title("Correlation Matrix")
    plt.xlabel("")
    plt.ylabel("")

    # Save the plot
    if save_plot:
        plt.savefig(os.path.join(PLOT_OUTPUT_FOLDER, "bias_correlation.pdf"), format='pdf', bbox_inches='tight')

    # Display the heatmap
    plt.show()


def plot_correlation_matrix_with_dendrogram(df: pd.DataFrame, save_plot: bool = True):
    """
    Plots a correlation matrix heatmap with a connected dendrogram for all numeric columns in the DataFrame.
    """

    # Step 1: Select all numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Step 2: Calculate the correlation matrix
    correlation_matrix = df[numeric_columns].corr()

    # Step 3: Plot the correlation matrix using seaborn's heatmap
    plt.figure(figsize=(8, 6.5))
    # sns.heatmap(correlation_matrix, annot=False, fmt=".1f", cmap=sns.diverging_palette(220, 20, as_cmap=True), center=0, cbar=True)

    g = sns.clustermap(correlation_matrix, annot=False, fmt=".1f", cmap=sns.diverging_palette(220, 20, as_cmap=True), center=0, cbar=True)
    g.ax_row_dendrogram.remove()
    
    # Step 4: Set plot title and labels
    plt.title("Correlation Matrix")
    plt.xlabel("")
    plt.ylabel("")

    # Save the plot
    if save_plot:
        plt.savefig(os.path.join(PLOT_OUTPUT_FOLDER, "bias_correlation.pdf"), format='pdf', bbox_inches='tight')

    # Display the heatmap
    plt.show()


def plot_bubble_plot(df: pd.DataFrame, x: str, y: str, size: str, color: str, label: str, xlim: tuple[float] = None, ylim: tuple[float] = None, legendloc: str = 'lower right', alpha: float = 1.0, label_offset: dict = {}, save_plot: bool = True):
    """
    Creates a bubble plot with the x and y axis representing the given columns, bubble size based on the 'size' column, and bubble color based on the 'color' column.
    """

    # Ensure the color column is treated as categorical
    df[color] = df[color].astype('category')

    # Get unique colors and assign a consistent color for each category
    unique_colors = df[color].cat.categories
    color_mapping = {cat: idx for idx, cat in enumerate(unique_colors)}

    # Create a colormap
    cmap = plt.get_cmap('Spectral', len(unique_colors))
    
    # Create a color list to ensure consistent colors for both plot and legend
    color_list = [cmap(color_mapping[cat]) for cat in df[color]]

    # Create the bubble plot
    plt.figure(figsize=(7, 5.5))
    scatter = plt.scatter(df[x], df[y], s=df[size]*10, c=color_list, alpha=alpha, edgecolor='lightgrey', linewidth=0.5)

    # Add labels to each bubble
    for i, row in df.iterrows():
        offset = label_offset[row[label]] if row[label] in label_offset else (0.0, 0.0)
        plt.text(row[x] + offset[0], row[y] + offset[1], str(row[label]), fontsize=9, ha='center', va='center', color='black')

    # Create a legend with categorical labels, using the same colormap
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / len(unique_colors)), linestyle='', markersize=10, markeredgecolor='lightgrey', markeredgewidth=0.5, alpha=alpha) for i in range(len(unique_colors))]
    plt.legend(handles, unique_colors, title=color, loc=legendloc)

    # Set plot labels and title
    plt.xlabel(x)
    plt.ylabel(y)

    # Manually set the axis limits, if provided
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    # Save the plot
    if save_plot:
        plt.savefig(os.path.join(PLOT_OUTPUT_FOLDER, "bubble_plot.pdf"), format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()


def compute_delta_scores(
    df_conditional: pd.DataFrame,
    df_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-test-case delta scores: conditional_score − baseline_score.

    The baseline should be the neutral condition (human_bias="neutral") from
    conditional_test_decision.py, giving a within-framework reference that
    controls for the mere presence of a human turn.

    Args:
        df_conditional: Results from conditional runs (any non-neutral human_bias).
        df_baseline: Results from the neutral baseline run (human_bias="neutral").

    Returns:
        DataFrame with columns: id, bias, human_bias, model, individual_score,
        baseline_score, delta_score.
    """
    base = (
        df_baseline[df_baseline["status"] == "OK"][["id", "bias", "model", "individual_score"]]
        .rename(columns={"individual_score": "baseline_score"})
    )
    cond = df_conditional[df_conditional["status"] == "OK"][
        ["id", "bias", "human_bias", "model", "individual_score"]
    ]
    merged = cond.merge(base, on=["id", "bias", "model"], how="inner")
    merged["delta_score"] = merged["individual_score"] - merged["baseline_score"]
    return merged


def compute_bias_similarity_matrix(
    biases: list[str],
    cache_path: str = None,
) -> pd.DataFrame:
    """
    Compute pairwise semantic similarity between bias names using OpenAI embeddings.

    Embeds each bias name as "<name> cognitive bias" with text-embedding-3-small,
    then returns the cosine-similarity matrix. Results are cached to avoid
    redundant API calls.

    Args:
        biases: List of bias names (formatted, e.g. "Framing Effect").
        cache_path: Optional path to a JSON file used for caching embeddings.
                    Defaults to data/bias_embeddings.json relative to analysis.py.

    Returns:
        Square DataFrame (index = columns = biases) with cosine similarity values in [−1, 1].
    """
    import json
    from openai import OpenAI

    if cache_path is None:
        cache_path = os.path.join("..", "data", "bias_embeddings.json")

    embeddings: dict[str, list[float]] = {}

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            embeddings = json.load(f)

    missing = [b for b in biases if b not in embeddings]
    if missing:
        client = OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[f"{b} cognitive bias" for b in missing],
        )
        for bias, item in zip(missing, response.data):
            embeddings[bias] = item.embedding
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(embeddings, f)

    vectors = np.array([embeddings[b] for b in biases])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = (vectors / norms) @ (vectors / norms).T
    return pd.DataFrame(similarity, index=biases, columns=biases)


def plot_hypothesis_diagnostics(
    df_delta: pd.DataFrame,
    similarity_matrix: pd.DataFrame = None,
    model: str = None,
    figsize: tuple = (20, 6),
    save_plot: bool = True,
):
    """
    Three-panel diagnostic figure for adjudicating between bias-induction hypotheses.

    Panel 1 — Delta heatmap
        Rows = target bias, columns = human bias, values = mean(delta_score).
        Flat/uniform → sycophancy. Strong diagonal → style mirroring.
        Structured off-diagonal → latent priming.

    Panel 2 — Diagonal vs. off-diagonal distributions
        Violin / strip plot comparing delta_score for on-diagonal pairs (target == human)
        against off-diagonal pairs. Style mirroring predicts diagonal >> off-diagonal.
        Sycophancy predicts diagonal ≈ off-diagonal.

    Panel 3 — Semantic similarity correlation
        Scatter plot of (semantic similarity, delta_score) for all (target, human) pairs.
        A positive slope supports latent priming; no slope supports style mirroring or
        sycophancy depending on Panel 2.

    Args:
        df_delta: Output of compute_delta_scores().
        similarity_matrix: Output of compute_bias_similarity_matrix(). If None,
                           Panel 3 is omitted.
        model: Optional model name for the figure title.
        figsize: Figure size.
        save_plot: If True, save as PDF.
    """
    n_panels = 3 if similarity_matrix is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    title_suffix = f" — {model}" if model else ""

    # ── Panel 1: Delta heatmap ──────────────────────────────────────────────
    ax1 = axes[0]
    heatmap_data = df_delta.pivot_table(
        values="delta_score",
        index="bias",
        columns="human_bias",
        aggfunc="mean",
    )
    heatmap_data = heatmap_data.sort_values(by=heatmap_data.columns[0], ascending=False)

    sns.heatmap(
        heatmap_data.round(2),
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        annot=True,
        fmt=".2f",
        vmin=-1.0,
        vmax=1.0,
        cbar=True,
        linewidths=0,
        ax=ax1,
    )
    ax1.set_title(f"Δ Score (conditional − baseline){title_suffix}", pad=10)
    ax1.set_xlabel("Human Bias", labelpad=8)
    ax1.set_ylabel("Target Bias", labelpad=8)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="left")
    plt.setp(ax1.get_yticklabels(), rotation=0)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # ── Panel 2: Diagonal vs. off-diagonal ─────────────────────────────────
    ax2 = axes[1]
    df_delta = df_delta.copy()
    df_delta["pair_type"] = np.where(
        df_delta["bias"] == df_delta["human_bias"], "On-diagonal\n(target = human)", "Off-diagonal\n(target ≠ human)"
    )

    sns.violinplot(
        data=df_delta,
        x="pair_type",
        y="delta_score",
        palette={"On-diagonal\n(target = human)": "#4393c3",
                 "Off-diagonal\n(target ≠ human)": "#d6604d"},
        inner="box",
        ax=ax2,
    )
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_title(f"Style mirroring vs. sycophancy{title_suffix}", pad=10)
    ax2.set_xlabel("")
    ax2.set_ylabel("Δ Score")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    # Annotate with means
    for i, ptype in enumerate(["On-diagonal\n(target = human)", "Off-diagonal\n(target ≠ human)"]):
        mean_val = df_delta[df_delta["pair_type"] == ptype]["delta_score"].mean()
        ax2.text(i, mean_val + 0.03, f"μ={mean_val:.2f}", ha="center", fontsize=9, color="black")

    # ── Panel 3: Similarity correlation (latent priming) ───────────────────
    if similarity_matrix is not None:
        ax3 = axes[2]

        # Build a long-form similarity table
        sim_long = similarity_matrix.stack().reset_index()
        sim_long.columns = ["bias", "human_bias", "similarity"]

        # Merge with mean delta scores per (target, human) pair
        mean_delta = df_delta.groupby(["bias", "human_bias"])["delta_score"].mean().reset_index()
        merged = mean_delta.merge(sim_long, on=["bias", "human_bias"], how="inner")

        # Exclude identical pairs (diagonal) from this panel since similarity=1 trivially
        merged = merged[merged["bias"] != merged["human_bias"]]

        ax3.scatter(
            merged["similarity"],
            merged["delta_score"],
            alpha=0.6,
            s=30,
            color="#666666",
        )

        # Regression line
        if len(merged) > 1:
            m_reg, b_reg = np.polyfit(merged["similarity"], merged["delta_score"], 1)
            x_line = np.linspace(merged["similarity"].min(), merged["similarity"].max(), 100)
            ax3.plot(x_line, m_reg * x_line + b_reg, color="#e63946", linewidth=1.5, label=f"slope={m_reg:.2f}")
            corr = merged["similarity"].corr(merged["delta_score"])
            ax3.text(
                0.05, 0.93, f"r = {corr:.2f}", transform=ax3.transAxes,
                fontsize=10, va="top", color="#e63946"
            )

        ax3.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax3.set_title(f"Latent priming: similarity vs. Δ{title_suffix}", pad=10)
        ax3.set_xlabel("Semantic similarity (human ↔ target bias)")
        ax3.set_ylabel("Δ Score")
        for spine in ["top", "right"]:
            ax3.spines[spine].set_visible(False)

    plt.tight_layout()

    if save_plot:
        os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
        fname = "hypothesis_diagnostics"
        if model:
            fname += f"_{model.replace(' ', '_')}"
        plt.savefig(
            os.path.join(PLOT_OUTPUT_FOLDER, f"{fname}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()


def load_conditional_data(
    folder_path: str = CONDITIONAL_RESULTS_FOLDER,
    model: str = None,
    format: bool = True,
) -> pd.DataFrame:
    """
    Loads conditional decision results produced by conditional_test_decision.py.

    Args:
        folder_path: Directory containing per-model CSV files.
        model: If provided, load only the CSV for this model name (e.g. "GPT-4o-Mini").
                If None, load and concatenate all model CSVs found in the folder.
        format: If True, apply the same bias / model name formatting as load_decision_data.

    Returns:
        A DataFrame with columns including bias, human_bias, model, individual_score, etc.
    """
    dataframes = []
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".csv"):
            continue
        if model is not None and file_name != f"{model}.csv":
            continue
        df = pd.read_csv(os.path.join(folder_path, file_name))
        dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(
            f"No conditional result CSV files found in '{folder_path}'"
            + (f" for model '{model}'" if model else "")
        )

    df = pd.concat(dataframes, ignore_index=True)

    if format:
        df = format_decision_data(df, format_bias_names=True, format_model_names=True)
        # Apply the same formatting to human_bias column
        if "human_bias" in df.columns:
            df["human_bias"] = df["human_bias"].apply(
                lambda x: re.sub(r'([a-z])([A-Z])', r'\1 \2', str(x))
            ).replace(BIAS_NAME_MAPPING)

    return df


def plot_conditional_bias_heatmap(
    df: pd.DataFrame = None,
    folder_path: str = CONDITIONAL_RESULTS_FOLDER,
    model: str = None,
    agg: str = "mean",
    abs: bool = False,
    figsize: tuple = (14, 10),
    save_plot: bool = True,
):
    """
    Plots a heatmap of mean bias scores with target bias on the Y-axis and human bias
    (the bias expressed by the simulated human interlocutor) on the X-axis.

    Each cell shows the mean individual_score for test cases where the LLM was evaluated
    on (row) target bias after being exposed to (column) human bias in the conversation.

    Args:
        df: Pre-loaded conditional results DataFrame. If None, loaded from folder_path.
        folder_path: Directory to load results from (used only when df is None).
        model: If provided, filter to a single model.
        agg: Aggregation function for pivot_table ('mean', 'median', etc.).
        abs: If True, use absolute values of individual_score.
        figsize: Figure size as (width, height).
        save_plot: If True, save the plot as a PDF.
    """
    if df is None:
        df = load_conditional_data(folder_path=folder_path, model=model)
    elif model is not None:
        df = df[df["model"] == model].copy()

    # Drop rows without a computed score
    df = df.dropna(subset=["individual_score"])

    if abs:
        df = df.copy()
        df["individual_score"] = df["individual_score"].abs()

    # Build the pivot table: rows = target bias, columns = human bias
    heatmap_data = df.pivot_table(
        values="individual_score",
        index="bias",
        columns="human_bias",
        aggfunc=agg,
    )

    # Add row-wise and column-wise averages
    heatmap_data["Average"] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by="Average", ascending=False)
    heatmap_data.loc["Average"] = heatmap_data.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data.round(2),
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        annot=True,
        fmt=".2f",
        vmin=-1.0,
        vmax=1.0,
        cbar=False,
        linewidths=0,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Human Bias", labelpad=12)
    ax.set_ylabel("Target Bias", labelpad=12)

    title = "Conditional Bias Scores"
    if model:
        title += f" — {model}"
    ax.set_title(title, pad=16)

    # Labels on top for X-axis; rotate for readability
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xticks(rotation=45, ha="left")
    plt.yticks(rotation=0)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # White separator line before the Average row/column
    ax.hlines([heatmap_data.shape[0] - 1], *ax.get_xlim(), color="white", linewidth=5)
    ax.vlines([heatmap_data.shape[1] - 1], *ax.get_ylim(), color="white", linewidth=5)

    # Colorbar below the heatmap
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.025])
    cbar = plt.colorbar(ax.collections[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mean Bias Score")
    cbar.outline.set_visible(False)

    plt.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.10)

    if save_plot:
        os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
        fname = "conditional_bias_heatmap"
        if model:
            fname += f"_{model.replace(' ', '_')}"
        plt.savefig(
            os.path.join(PLOT_OUTPUT_FOLDER, f"{fname}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()
