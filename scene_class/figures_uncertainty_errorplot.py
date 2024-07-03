import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the CSV file
csv_file = '/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/data/results/pretrained_BB/EfficientNet_b4.csv'

# Read the CSV file into a pandas dataframe
df = pd.read_csv(csv_file)

correct_predictions = True
if correct_predictions:
    df = df[df['labels'] == df['prediction']]


# family_names = df["family"].unique()
# short_family_names = [(f[:3] + '.') for f in family_names]
# print(short_family_names)


# Print the dataframe
print(df.head())

# Create the box plot
sns.boxplot(x='family', y='probabilities', data=df, color='#2a6f97', linecolor='#013a63', linewidth=0.5, whis=(0, 100))
plt.xlabel('Family')
plt.ylabel('Probabilities')
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.5)
plt.savefig('/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/paper/figures/uncertainty_efficientnet/uncertainty_efficientnet.png', dpi=300)
plt.show()

tryd=1