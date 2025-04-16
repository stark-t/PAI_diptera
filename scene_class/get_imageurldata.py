import pandas as pd
import os
import json

# Define the file paths
file_path = r"D:/2023_PAI_diptera/data/df_sample_20230213.txt"
json_dir = r"D:/2023_PAI_diptera/data/json_files"
output_file_path = r"D:/2023_PAI_diptera/data/PAI_Diptera_family_GBIF"

# Read the text file as a DataFrame
df = pd.read_csv(file_path, sep='\t')  # Adjust the separator if needed (e.g., ',' or '\t')
# print(df.head())
print("DataFrame Columns:")
for column in df.columns:
    print(f"- {column}")

# Print the first 5 entries from the 'file_name' column
print("First 5 entries from 'file_name':")
print(df['file_name'].head())


# Loop through each file in the directory
for file_name in os.listdir(json_dir):
    if file_name.endswith('.json'):  # Check if the file is a JSON file
        file_path = os.path.join(json_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            file_id_list = data['_via_image_id_list']
            # print(file_id_list[0:5])
        
        # Check if the file_id_list is in the DataFrame
        json_file_names = [f.split('.')[0] for f in file_id_list]
        matching_entries = df[df['file_name'].isin(json_file_names)]

        # Filter the matching entries to include only the specified columns
        filtered_entries = matching_entries[['id', 'gbifID', 'file_name', 'license']]
        
        # Export the filtered entries to a CSV file
        output_file_path_family = os.path.join(output_file_path, f"{file_name.split('.')[0]}.csv")
        with open(output_file_path_family, 'w', encoding='utf-8') as output_file:
            filtered_entries.to_csv(output_file, index=False)
        print(f"Filtered entries exported to {output_file_path}")
        d=1
            