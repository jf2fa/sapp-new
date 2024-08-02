import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
import os

# Define the CSV path variables
example_csv_path = 'synthetic_survey_responses_1.csv'
output_csv_path = 'nmssynthetic_survey_responses_1.csv'
num_rows = 100  # Number of rows of synthetic data to generate
description_file = 'description.json'  # Temporary file to store the data description
temp_csv_path = 'temp_example.csv'  # Temporary CSV file path

# Function to generate synthetic data based on example CSV
def generate_synthetic_data(example_csv_path, output_csv_path, num_rows=100):
    try:
        # Load the example CSV
        df_example = pd.read_csv(example_csv_path)
        
        # Save the DataFrame to a temporary CSV file
        df_example.to_csv(temp_csv_path, index=False)

        # Describe the data
        describer = DataDescriber(category_threshold=50)
        describer.describe_dataset_in_correlated_attribute_mode(temp_csv_path, k=2)
        describer.save_dataset_description_to_file(description_file)

        # Generate synthetic data
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_rows, description_file)
        synthetic_data = generator.synthetic_dataset

        # Save the synthetic data to a CSV file
        synthetic_data.to_csv(output_csv_path, index=False)
        print(f"Synthetic data generated and saved to {output_csv_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        if os.path.exists(description_file):
            os.remove(description_file)

# Generate synthetic data
generate_synthetic_data(example_csv_path, output_csv_path, num_rows)
