import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from confusion_matrix import cm_analysis

# Path to the CSV file
csv_file_BB = '/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/data/results/pretrained_BB/EfficientNet_b4.csv'
csv_file_noBB = '/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/data/results/pretrained_noBB/EfficientNet_b4.csv'

# Read the CSV file into a pandas dataframe
df_BB = pd.read_csv(csv_file_BB)
df_noBB = pd.read_csv(csv_file_noBB)

# Print the dataframe
print(df_BB.head())
family_names = df_BB["family"].unique()
short_family_names = [(f[:3] + '.') for f in family_names]
print(short_family_names)

cm_BB = cm_analysis(df_BB["labels"], df_BB["prediction"], short_family_names, figsize=(17, 17), plot=True, fontsize=22, 
                    filename='/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/paper/figures/confusion_matrix/BB_EfficientNet_b4.png',
                    filename_tex='/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/paper/figures/confusion_matrix/BB_EfficientNet_b4.tex')

cm_npBB = cm_analysis(df_noBB["labels"], df_noBB["prediction"], short_family_names, figsize=(17, 17), plot=True, fontsize=22, 
                      filename='/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/paper/figures/confusion_matrix/noBB_EfficientNet_b4.png',
                      filename_tex='/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/paper/figures/confusion_matrix/noBB_EfficientNet_b4.tex')

