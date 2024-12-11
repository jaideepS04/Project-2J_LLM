import pandas as pd
import openai
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up the API key and proxy URL
openai.api_key = os.getenv("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDI0ODJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.AoGx1fHFJVPj4Lu5O5uTM7gM_JT8m_9RbYSEjRdOhcI")
 # Secure setup (best practice)
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# List of CSV files to process
CSV_FILES = ["goodreads.csv", "happiness.csv","media.csv"]

# Process Each CSV File
for csv_file in CSV_FILES:
    print(f"Processing {csv_file}...")
    
    try:
        # Try reading the dataset with different encoding
        data = pd.read_csv(csv_file, encoding='ISO-8859-1')  # Try 'ISO-8859-1' or 'latin1'
    except UnicodeDecodeError:
        print(f"Error: Unable to read {csv_file} due to encoding issues.")
        continue  # Skip to the next file if encoding fails

    print(data.columns)
    print(data.head())  # Check the first few rows

    summary = data.describe(include="all")
    print(summary)

    missing_data = data.isnull().sum()
    print(missing_data)

    # AI Analysis - Generate summary for each CSV file
    report_prompt = f"""
    Create a summary of the dataset {csv_file}. Include:
    - Overview of key statistics
    - Top trends or correlations
    - Recommendations for next steps
    """

    openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDI0ODJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.AoGx1fHFJVPj4Lu5O5uTM7gM_JT8m_9RbYSEjRdOhcI" # Replace this with your real token
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[ 
            {"role": "system", "content": "You are an assistant generating data analysis summaries."},
            {"role": "user", "content": report_prompt}
        ],
        max_tokens=300
    )
    
    report_text = response['choices'][0]['message']['content'].strip()

    # Save the report for each dataset
    report_filename = f"{os.path.splitext(csv_file)[0]}_README.md"
    with open(report_filename, "w") as f:
        f.write(f"# Data Analysis Report for {csv_file.capitalize()}\n")
        f.write("## Summary\n")
        f.write(report_text + "\n\n")

    # Visualization: Top Genres or Any Relevant Columns (if applicable)
    if "Genre" in data.columns and "Rating" in data.columns:
        top_genres = data.groupby("Genre")["Rating"].mean().sort_values(ascending=False).head(5)
        sns.barplot(x=top_genres.index, y=top_genres.values)
        plt.title(f"Top 5 Genres by Average Rating for {csv_file}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the bar plot for each dataset
        output_path = f"{os.path.splitext(csv_file)[0]}_top_genres.png"
        plt.savefig(output_path)
        plt.clf()  # Clear the plot for the next iteration
    else:
        print(f"Skipping top genres visualization for {csv_file} - 'Genre' or 'Rating' column missing.")    

    # Correlation Heatmap for numeric columns (if applicable)
    numeric_data = data.select_dtypes(include=['number'])
    if not numeric_data.empty:
        corr = numeric_data.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title(f"Correlation Heatmap for {csv_file}")
        plt.tight_layout()
        
        # Save the heatmap for each dataset
        heatmap_filename = f"{os.path.splitext(csv_file)[0]}_heatmapnew.png"
        plt.savefig(heatmap_filename)
        plt.clf()  # Clear the plot for the next iteration

    print(f"Finished processing {csv_file}. Results saved for {csv_file}.\n")

print("Processing complete for all datasets.")