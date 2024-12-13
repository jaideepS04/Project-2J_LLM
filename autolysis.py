import pandas as pd
import openai
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Secure setup: Load API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Ensure API key is set
if not openai.api_key:
    raise ValueError("OpenAI API key not set. Please configure the environment variable 'OPENAI_API_KEY'.")

# List of CSV files to process
CSV_FILES = ["goodreads.csv", "happiness.csv", "media.csv"]

# Function to generate a dynamic prompt based on the dataset structure
# The code generates dynamic prompts based on the dataset's structure and content.
def generate_dynamic_prompt(csv_file, data):
    # Check for relevant columns
    prompts = [f"Analyze the dataset {csv_file}. Provide insights on the following:"]
    
    if "Rating" in data.columns:
        prompts.append("- Analyze the distribution of ratings, and any correlations with other variables.")
    if "Genre" in data.columns:
        prompts.append("- Analyze the distribution of genres and their relationship with ratings or other variables.")
    if "Date" in data.columns:
        prompts.append("- Explore any temporal trends in the data (e.g., ratings over time).")
    
    # Add missing data analysis if applicable
    missing_data = data.isnull().sum()
    if missing_data.any():
        prompts.append("- Provide recommendations for handling missing data.")
    
    # Return the dynamically constructed prompt as a single string
    return " ".join(prompts)

# Process each CSV file
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

    # Generate descriptive statistics
    summary = data.describe(include="all")
    print(summary)

    # Count missing values
    missing_data = data.isnull().sum()
    print("Missing data:", missing_data)

    # Dynamic AI Analysis: Generate summary for each CSV file based on its content
    report_prompt = generate_dynamic_prompt(csv_file, data)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[ 
                {"role": "system", "content": "You are an assistant generating data analysis summaries."},
                {"role": "user", "content": report_prompt}
            ],
            max_tokens=500
        )
        report_text = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating AI report for {csv_file}: {e}")
        report_text = "Error generating report. Please check the dataset and API configuration."

    # Save the report for each dataset
    report_filename = f"{os.path.splitext(csv_file)[0]}_README.md"
    with open(report_filename, "w") as f:
        f.write(f"# Data Analysis Report for {csv_file.capitalize()}\n")
        f.write("## Summary\n")
        f.write(report_text + "\n\n")

    # Visualization: Top Genres or Any Relevant Columns (if applicable)
    if "Genre" in data.columns and "Rating" in data.columns:
        try:
            top_genres = data.groupby("Genre")["Rating"].mean().sort_values(ascending=False).head(5)
            sns.barplot(x=top_genres.index, y=top_genres.values)
            plt.title(f"Top 5 Genres by Average Rating for {csv_file}")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the bar plot for each dataset
            output_path = f"{os.path.splitext(csv_file)[0]}_top_genres.png"
            plt.savefig(output_path)
            plt.clf()  # Clear the plot for the next iteration
        except Exception as e:
            print(f"Error generating genre visualization for {csv_file}: {e}")
    else:
        print(f"Skipping top genres visualization for {csv_file} - 'Genre' or 'Rating' column missing.")    

    # Correlation Heatmap for numeric columns (if applicable)
    numeric_data = data.select_dtypes(include=['number'])
    if not numeric_data.empty:
        try:
            corr = numeric_data.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title(f"Correlation Heatmap for {csv_file}")
            plt.tight_layout()

            # Save the heatmap for each dataset
            heatmap_filename = f"{os.path.splitext(csv_file)[0]}_heatmap.png"
            plt.savefig(heatmap_filename)
            plt.clf()  # Clear the plot for the next iteration
        except Exception as e:
            print(f"Error generating heatmap for {csv_file}: {e}")

    # Dynamic Prompt for More Specific Analysis: Detect outliers or transformations
    outlier_prompt = f"""
    Based on the dataset {csv_file}, identify any potential outliers in the numerical data and suggest transformations or cleaning steps.
    """
    try:
        outlier_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant focused on advanced data analysis."},
                {"role": "user", "content": outlier_prompt}
            ],
            max_tokens=500
        )
        outlier_text = outlier_response['choices'][0]['message']['content'].strip()
        with open(report_filename, "a") as f:
            f.write("## Advanced Insights: Outlier Analysis\n")
            f.write(outlier_text + "\n\n")
    except Exception as e:
        print(f"Error generating advanced insights for outliers in {csv_file}: {e}")

    print(f"Finished processing {csv_file}. Results saved for {csv_file}.\n")

print("Processing complete for all datasets.")
