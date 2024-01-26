import os
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Get all the csv files in the results folder
csv_files = glob.glob('data/results/*.csv')

# Create an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Loop through each CSV file
for file in csv_files:
    # Load the CSV file as a pandas DataFrame
    df = pd.read_csv(file)
    
    # Get the model name from the first item of the modelname_probability.csv file
    model_name = file.split(os.sep)[-1].split('_')[0]
    
    #delete the last row of the dataframe if the model is stnet
    if model_name == 'stnet':
        df = df[:-1]
    
    if model_name == 'stnet':
        model_name = 'STNet'
    elif model_name == 'efficientnetb4':
        model_name = 'EfficientNet_b4'
    elif model_name == 'seresnext32x4d':
        model_name = 'SeResNeXt_32x4d'
    elif model_name == 'resnet18':
        model_name = 'ResNet-18'
    elif model_name == 'mobilenetv3large':
        model_name = 'MobileNetV3_large'

    # Add a column with the model name for each row
    df['model_name'] = model_name
    
    # Concatenate the current DataFrame with the combined DataFrame
    combined_df = pd.concat([combined_df, df], ignore_index=True)
    

    
print(combined_df.head())


# # Create a grouped error boxplot using Seaborn
# plt.figure(figsize=(23, 8))
# sns.set(style="whitegrid", font_scale=1.5)  # Set the font scale to 1.5
# hue_order = ['STNet', 'MobileNetV3_large', 'ResNet-18', 'SeResNeXt_32x4d', 'EfficientNet_b4']
# sns.boxplot(x="family", y="probabilities", hue="model_name", data=combined_df, palette="muted", 
#             dodge=True, gap=0.2, linewidth=0.5, fliersize=3,
#             hue_order=hue_order)
# plt.xlabel("Family", fontsize=16)  # Increase the font size of the x-axis label
# plt.ylabel("Probabilities", fontsize=16)  # Increase the font size of the y-axis label
# plt.legend(loc='lower left', fontsize=16)  # Increase the font size of the legend
# plt.xticks(fontsize=12)  # Increase the font size of the x-axis tick labels
# plt.yticks(fontsize=16)  # Increase the font size of the y-axis tick labels
# plt.savefig(os.path.join("D:\\2023_PAI_diptera\\PAI_diptera\\data\\results", "boxplot_allCNNs.png"), dpi=300)
# plt.show()

# Filter the combined_df based on labels == prediction
df = combined_df[combined_df['labels'] == combined_df['prediction']]

# # Create a grouped error boxplot using Seaborn
# plt.figure(figsize=(23, 8))ST
# sns.set(style="whitegrid", font_scale=1.5)  # Set the font scale to 1.5
# hue_order = ['STNet', 'MobileNetV3_large', 'ResNet-18', 'SeResNeXt_32x4d', 'EfficientNet_b4']
# sns.boxplot(x="family", y="probabilities", hue="model_name", data=filtered_df, palette="muted", 
#             dodge=True, gap=0.2, linewidth=0.5, fliersize=3,
#             hue_order=hue_order)
# plt.xlabel("Family", fontsize=16)  # Increase the font size of the x-axis label
# plt.ylabel("Probabilities", fontsize=16)  # Increase the font size of the y-axis label
# plt.legend(loc='lower left', fontsize=16)  # Increase the font size of the legend
# plt.xticks(fontsize=12)  # Increase the font size of the x-axis tick labels
# plt.yticks(fontsize=16)  # Increase the font size of the y-axis tick labels
# plt.savefig(os.path.join("D:\\2023_PAI_diptera\\PAI_diptera\\data\\results", "boxplot_allCNNs_correct.png"), dpi=300)
# plt.show()

# Calculate the mean probability per class for all model_names
mean_probabilities = df.groupby(['model_name', 'family'])['probabilities'].mean()

# Pivot the mean_probabilities DataFrame
pivot_table = mean_probabilities.reset_index().pivot(index='model_name', columns='family', values='probabilities')

# Convert the pivot table values to xx.xx% formatSt
pivot_table_humanreadable = pivot_table.applymap(lambda x: f'{x:.2%}')

# Print the pivot table
print(pivot_table_humanreadable)

# Export the pivot table as a LaTeX-style table to a .txt file
with open(os.path.join("D:\\2023_PAI_diptera\\PAI_diptera\\data\\results", 'pivot_table.txt'), 'w') as file:
    file.write(pivot_table_humanreadable.to_latex())

# Calculate the mean probability per class for all model_names
mean_probabilities_all = df.groupby(['model_name'])['probabilities'].mean()
print(mean_probabilities_all.apply(lambda x: f'{x:.2%}'))

# # Create a confidence matrix
# # Create a 2x2 matrix where "Confident Correct", "Confident Incorrect", "Not Confident Correct", and "Not Confident Incorrect" are the classes
# # Filter the dataframe based on the conditions
# matrix_confident_correct = df[(df["labels"] == df["prediction"]) & (df["probabilities"] > 0.5)].shape[0]
# matrix_confident_incorrect = df[(df["labels"] != df["prediction"]) & (df["probabilities"] > 0.5)].shape[0]
# matrix_not_confident_correct = df[(df["labels"] == df["prediction"]) & (df["probabilities"] <= 0.5)].shape[0]
# matrix_not_confident_incorrect = df[(df["labels"] != df["prediction"]) & (df["probabilities"] <= 0.5)].shape[0]

# # Create a dataframe from the matrix
# matrix_data = {
#     "Confident Correct": [matrix_confident_correct],
#     "Confident Incorrect": [matrix_confident_incorrect],
#     "Not Confident Correct": [matrix_not_confident_correct],
#     "Not Confident Incorrect": [matrix_not_confident_incorrect]
# }
# matrix_df = pd.DataFrame(matrix_data)
# print(matrix_df)

# # Create a seaborn heatmap of the confidence matrix
# sns.heatmap(matrix_df, annot=True, fmt="d", cmap="binary", cbar=False)
# plt.title("Confidence Matrix")
# plt.show()

EfficientNet_b4_df = combined_df[combined_df['model_name'] == 'EfficientNet_b4']

# Create a heatmap of the confusion matrix
plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)  # Set the font scale to 1.5
sns.heatmap(pd.crosstab(EfficientNet_b4_df['labels'], EfficientNet_b4_df['prediction'], normalize='index'), annot=True, fmt=".2%", cmap="binary", cbar=False)
plt.title("Confusion Matrix")
plt.show()


