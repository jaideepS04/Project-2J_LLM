import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions" 
MODEL = "gpt-4o-mini"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN environment variable not set.")

def send_to_openai(prompt, detail="default"):
    """Send a prompt to the OpenAI API and return the response."""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Summarize the data analysis."},
                {"role": "user", "content": prompt}
            ]
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def summarize_data_overview(df):
    """Summarize the data's basic overview."""
    info = df.info(buf=None)
    description = df.describe().to_string()
    nulls = df.isnull().sum().to_string()
    summary = f"Data Info:\n{info}\n\nData Description:\n{description}\n\nMissing Values:\n{nulls}"
    return send_to_openai(summary)

def clean_missing_data(df):
    """Clean missing data by dropping columns with high percentage of missing values and rows with any remaining missing values."""
    threshold = 0.5
    df_cleaned = df.dropna(axis=1, thresh=len(df) * (1 - threshold))
    df_cleaned = df_cleaned.dropna()
    return df_cleaned

def detect_outliers_and_anomalies(df):
    """Detect outliers and anomalies in numerical columns."""
    results = []
    for column in df.select_dtypes(include=['number']):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        min_val = q1 - 1.5 * iqr
        max_val = q3 + 1.5 * iqr
        results.append(f"{column}: Q1={q1}, Q3={q3}, IQR={iqr}, Min={min_val}, Max={max_val}")
    return send_to_openai("\n".join(results))

def compute_correlation_summary(df):
    """Compute the correlation matrix and summarize key insights."""
    try:
        numerical_df = df.select_dtypes(include=["number"])
        correlation = numerical_df.corr()
        correlation_str = correlation.to_string()
        prompt = f"Here is the correlation matrix:\n{correlation_str}\nSummarize the key insights."
        summary = send_to_openai(prompt)
        return summary
    except Exception as e:
        return f"Error in computing correlation: {e}"

def visualize_numerical_columns(df):
    """Visualize numerical columns using pair plots."""
    numerical_cols = df.select_dtypes(include=['number'])
    img_paths = []
    if not numerical_cols.empty:
        # Ensure there are valid numerical columns with non-NaN values
        valid_cols = numerical_cols.dropna(axis=1, how='all')
        if not valid_cols.empty:
            plt.figure(figsize=(10, 10), dpi=100)
            sns.pairplot(valid_cols)
            img_path = "numerical_plot.png"
            plt.savefig(img_path)
            plt.close()
            img_paths.append(img_path)
            print(f"Numerical columns visualization saved as {img_path}.")
        else:
            print("No valid numerical columns with data found for visualization.")
    else:
        print("No numerical columns found for visualization.")
    
    return img_paths

def visualize_categorical_columns(df):
    """Visualize categorical columns using bar plots."""
    categorical_cols = df.select_dtypes(include=['object', 'category'])
    img_paths = []
    if not categorical_cols.empty:
        for col in categorical_cols.columns:
            unique_values_count = df[col].nunique()  # Count unique values in the column
            if unique_values_count > 30:
                print(f"Skipping {col} as it has {unique_values_count} unique values.")
                continue  # Skip columns with more than 30 unique values
            
            # Proceed to plot if unique values are <= 30
            plt.figure(figsize=(8, 8), dpi=100)
            sns.countplot(y=col, data=df, order=df[col].value_counts().index)
            img_path = f"{col}_plot.png"
            plt.title(f"Distribution of {col}")
            plt.savefig(img_path)
            plt.close()
            img_paths.append(img_path)
            print(f"Categorical column {col} visualization saved as {img_path}.")
    else:
        print("No categorical columns found for visualization.")
    
    return img_paths

def generate_summary_and_visualizations(df):
    """Generate a comprehensive data summary and visualize columns."""
    # Generate summary
    overview_summary = summarize_data_overview(df)
    outliers_summary = detect_outliers_and_anomalies(df)
    correlation_summary = compute_correlation_summary(df)
    
    # Visualize numerical and categorical columns
    numerical_visualizations = visualize_numerical_columns(df)
    categorical_visualizations = visualize_categorical_columns(df)
    
    # Return everything for further processing
    return {
        "overview_summary": overview_summary,
        "outliers_summary": outliers_summary,
        "correlation_summary": correlation_summary,
        "numerical_visualizations": numerical_visualizations,
        "categorical_visualizations": categorical_visualizations
    }
