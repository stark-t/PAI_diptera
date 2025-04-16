import os
import pandas as pd

directory = '/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/paper/figures/all_images'
df_all_efficientnet = pd.DataFrame()
for root, dirs, files in os.walk(directory):
    for dir in dirs:
        stats_file = os.path.join(root, dir, 'stats.csv')
        if os.path.isfile(stats_file):
            df = pd.read_csv(stats_file)
            efficientnetb4_columns = df[df['modelname'] == 'EfficientNet_b4']
            df_all_efficientnet = pd.concat([df_all_efficientnet, efficientnetb4_columns], ignore_index=True)
            print(df_all_efficientnet)
            d=1