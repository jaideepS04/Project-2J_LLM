import pandas as pd
import openai
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Secure setup: Load API key from environment variables
def setup_openai_api():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    if not openai.api_key:
        raise ValueError("OpenAI API key not set. Please configure the environment variable 'OPENAI_API_KEY'.")

# Function to generate a dynamic prompt based on the dataset structure
def generate_dynamic_prompt(csv_file, data):
    """Generates a dynamic prompt for the AI based on the dataset's columns."""
    prompts = [f"Analyze the dataset {csv_file}. Provide insights on the following:"]
    
    # The code makes use of dynamic prompts based on dataset content (such as Rating, Genre, Date)
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
    
    return " ".join(prompts)

# Function to read and process the CSV file
def load_csv_data(csv_file):
    """Loads CSV data with proper encoding handling."""
    try:
        data = pd.read_csv(csv_file, encoding='ISO-8859-1')  # Try 'ISO-8859-1' or 'latin1'
        return data
    except UnicodeDecodeError:
        print(f"Error: Unable to read {csv_file} due to encoding issues.")
        return None  # Return None if the file can't be read

# Function to generate a descriptive analysis report
def generate_ai_report(csv_file, data, report_prompt):
    """Generates an AI analysis report using OpenAI's GPT model."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[ 
                {"role": "system", "content": "You are an assistant generating data analysis summaries."},
                {"role": "user", "content": report_prompt}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating AI report for {csv_file}: {e}")
        return "Error generating report. Please check the dataset and API configuration."

# Function to save the report to a markdown file
def save_report(csv_file, report_text):
    """Saves the generated AI report to a markdown file."""
    report_filename = f"{os.path.splitext(csv_file)[0]}_README.md"
    with open(report_filename, "w") as f:
        f.write(f"# Data Analysis Report for {csv_file.capitalize()}\n")
        f.write("## Summary\n")
        f.write(report_text + "\n\n")
    return report_filename

# Function for generating visualizations (Top Genres and Correlation Heatmap)
def generate_visualizations(data, csv_file):
    """Generates visualizations (e.g., top genres, correlation heatmap) for the dataset."""
    # Visualization 1: Top Genres or Any Relevant Columns (if applicable)
    if "Genre" in data.columns and "Rating" in data.columns:
        try:
            top_genres = data.groupby("Genre")["Rating"].mean().sort_values(ascending=False).head(5)
            sns.barplot(x=top_genres.index, y=top_genres.values)
            plt.title(f"Top 5 Genres by Average Rating for {csv_file}")
            plt.xticks(rotation=45)
            plt.tight_layout()

            output_path = f"{os.path.splitext(csv_file)[0]}_top_genres.png"
            plt.savefig(output_path)
            plt.clf()  # Clear the plot for the next iteration
        except Exception as e:
            print(f"Error generating genre visualization for {csv_file}: {e}")
    
    # Visualization 2: Correlation Heatmap for numeric columns (if applicable)
    numeric_data = data.select_dtypes(include=['number'])
    if not numeric_data.empty:
        try:
            corr = numeric_data.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title(f"Correlation Heatmap for {csv_file}")
            plt.tight_layout()

            heatmap_filename = f"{os.path.splitext(csv_file)[0]}_heatmap.png"
            plt.savefig(heatmap_filename)
            plt.clf()  # Clear the plot for the next iteration
        except Exception as e:
            print(f"Error generating heatmap for {csv_file}: {e}")

# Function to analyze outliers and suggest transformations
def analyze_outliers(csv_file):
    """Generates a dynamic prompt for outlier analysis and generates the AI response."""
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
        return outlier_response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating advanced insights for outliers in {csv_file}: {e}")
        return "Error generating outlier analysis."

# Main Function to process the CSV files
def process_csv_files():
    CSV_FILES = ["goodreads.csv", "happiness.csv", "media.csv"]

    for csv_file in CSV_FILES:
        print(f"Processing {csv_file}...")
        
        # Load dataset
        data = load_csv_data(csv_file)
        if data is None:
            continue  # Skip to next file if there was an issue loading the data
        
        # Generate AI Report
        report_prompt = generate_dynamic_prompt(csv_file, data)
        report_text = generate_ai_report(csv_file, data, report_prompt)
        
        # Save AI Report
        report_filename = save_report(csv_file, report_text)
        
        # Generate Visualizations
        generate_visualizations(data, csv_file)
        
        # Generate Outlier Analysis
        outlier_text = analyze_outliers(csv_file)
        with open(report_filename, "a") as f:
            f.write("## Advanced Insights: Outlier Analysis\n")
            f.write(outlier_text + "\n\n")
        
        print(f"Finished processing {csv_file}. Results saved for {csv_file}.\n")

    print("Processing complete for all datasets.")

# Run the script
if __name__ == "__main__":
    setup_openai_api()  # Setup OpenAI API key
    process_csv_files()  # Process all CSV files
