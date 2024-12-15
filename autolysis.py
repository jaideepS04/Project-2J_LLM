import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet

# Constants
API_URL_22f3001519 = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN_22f3001519 = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDE1MTlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.wZ9-GZNl26PuInJWOOgtIHHhb16qD1ph1kLHjSqukOQ"

def load_data_22f3001519(file_path_22f3001519):
    """Load CSV data with encoding detection."""
    with open(file_path_22f3001519, 'rb') as f_22f3001519:
        result_22f3001519 = chardet.detect(f_22f3001519.read())
    encoding_22f3001519 = result_22f3001519['encoding']
    return pd.read_csv(file_path_22f3001519, encoding=encoding_22f3001519)

def analyze_data_22f3001519(df_22f3001519):
    """Perform basic data analysis."""
    numeric_df_22f3001519 = df_22f3001519.select_dtypes(include=['number']) 
    analysis_22f3001519 = {
        'summary': df_22f3001519.describe(include='all').to_dict(),
        'missing_values': df_22f3001519.isnull().sum().to_dict(),
        'correlation': numeric_df_22f3001519.corr().to_dict()
    }
    return analysis_22f3001519

def visualize_data_22f3001519(df_22f3001519):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns_22f3001519 = df_22f3001519.select_dtypes(include=['number']).columns
    for column_22f3001519 in numeric_columns_22f3001519:
        plt.figure()
        sns.histplot(df_22f3001519[column_22f3001519].dropna(), kde=True, color="blue", edgecolor="black")
        plt.title(f'Distribution of {column_22f3001519}')
        plt.xlabel(column_22f3001519)
        plt.ylabel("Frequency")
        plt.savefig(f'{column_22f3001519}_distribution.png')
        plt.close()

    # Generate a correlation heatmap if there are multiple numeric columns
    if len(numeric_columns_22f3001519) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix_22f3001519 = df_22f3001519[numeric_columns_22f3001519].corr()
        sns.heatmap(correlation_matrix_22f3001519, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()

def generate_narrative_22f3001519(analysis_22f3001519):
    """Generate narrative using LLM."""
    headers_22f3001519 = {
        'Authorization': f'Bearer {AIPROXY_TOKEN_22f3001519}',
        'Content-Type': 'application/json'
    }
    prompt_22f3001519 = (
        f"Based on this detailed data analysis, craft an engaging narrative. "
        f"Highlight key findings from the summary and correlation matrix: {analysis_22f3001519}"
    )
    data_22f3001519 = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt_22f3001519}]
    }
    try:
        response_22f3001519 = httpx.post(API_URL_22f3001519, headers=headers_22f3001519, json=data_22f3001519, timeout=30.0)
        response_22f3001519.raise_for_status()
        return response_22f3001519.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e_22f3001519:
        print(f"HTTP error occurred: {e_22f3001519}")
    except httpx.RequestError as e_22f3001519:
        print(f"Request error occurred: {e_22f3001519}")
    except Exception as e_22f3001519:
        print(f"An unexpected error occurred: {e_22f3001519}")
    return "Narrative generation failed due to an error."

def create_markdown_report_22f3001519(narrative_22f3001519):
    """Create a markdown report file."""
    content_22f3001519 = (
        "# Data Analysis Report\n\n"
        "## Narrative Analysis\n\n"
        f"{narrative_22f3001519}\n\n"
        "## Visualizations\n\n"
        "The following visualizations have been generated:\n\n"
        "- Distribution plots for numeric columns\n"
        "- Correlation heatmap (if applicable)\n"
        "\nPlease refer to the generated image files for detailed insights."
    )
    with open('README.md', 'w') as f_22f3001519:
        f_22f3001519.write(content_22f3001519)

def main_22f3001519(file_path_22f3001519):
    df_22f3001519 = load_data_22f3001519(file_path_22f3001519)
    analysis_22f3001519 = analyze_data_22f3001519(df_22f3001519)
    visualize_data_22f3001519(df_22f3001519)
    narrative_22f3001519 = generate_narrative_22f3001519(analysis_22f3001519)
    create_markdown_report_22f3001519(narrative_22f3001519)

if _name_ == "_main_":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main_22f3001519(sys.argv[1])
