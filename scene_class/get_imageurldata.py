import pandas as pd

# Define the file path
file_path = r"Y:\data\PAI_diptera\df_sample_20230213.txt"

# Read the text file as a DataFrame
df = pd.read_csv(file_path, sep='\t')  # Adjust the separator if needed (e.g., ',' or '\t')

# Print the head of the DataFrame
print(df.head())