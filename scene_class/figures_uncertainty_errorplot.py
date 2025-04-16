import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the CSV file
csv_file_BB = "/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/data/results/pretrained_BB/EfficientNet_b4.csv"
csv_file_noBB = "/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/data/results/pretrained_noBB/EfficientNet_b4.csv"
# Read the CSV file into a pandas dataframe
df = pd.read_csv(csv_file_BB)
df_noBB = pd.read_csv(csv_file_noBB)

df["Bounding Box"] = True
df_noBB["Bounding Box"] = False

# Concatenate df and df_noBB
df_BBs = pd.concat([df, df_noBB], ignore_index=True)
df = df_BBs

correct_predictions = True
if correct_predictions:
    df = df[df["labels"] == df["prediction"]]

median_probabilities = df.groupby(["family", "Bounding Box"])["probabilities"].median()
print(median_probabilities)
# family_names = df["family"].unique()
# short_family_names = [(f[:3] + '.') for f in family_names]
# print(short_family_names)


# Print the dataframe
print(df.head())

# Create the box plot
palette = ["#2a6f97", "#0096c7"]
sns.boxplot(
    x="family",
    y="probabilities",
    hue="Bounding Box",
    data=df,
    palette=palette,
    # color="#2a6f97",
    # linecolor="#013a63",
    gap=0.1,
    fliersize=0.2,
    linewidth=0.5,
    # whis=(0, 100),
)
plt.xlabel("Family")
plt.ylabel("Probabilities")
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.5)
plt.savefig(
    "/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/paper/figures/uncertainty_efficientnet/uncertainty_efficientnet.png",
    dpi=300,
)
plt.show()
