# Utilizing CNNs for Classification and Uncertainty Quantification for 15 Families of European Fly Pollinators

Stark, T., Wurm, M., Ștefan, V., Wolf, F., Taubenböck, H., & Knight, T. M. (2025). *Utilizing CNNs for classification and uncertainty quantification for 15 families of European fly pollinators*. In review. Soon to be available on bioRxiv.

---

## Abstract

Pollination is essential for maintaining biodiversity and ensuring food security, and in Europe it is primarily mediated by four insect orders (Coleoptera, Diptera, Hymenoptera, Lepidoptera). However, traditional monitoring methods are costly and time consuming. Although recent automation efforts have focused on butterflies and bees, flies, a diverse and ecologically important group of pollinators—have received comparatively little attention. In this study, we investigate the use of CNNs to classify 15 European pollinating fly families and quantify classification uncertainty. EfficientNetB4 achieved an accuracy of up to 95.61%, with cropped inputs (based on bounding boxes) improving both accuracy and confidence. This approach marks a significant step forward in automated pollinator monitoring.

---

## How to Use This Repository

### 1. Install Requirements

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download the Data

To download and prepare the dataset, run the provided script:

```bash
python ./scene_class/get_gbif_data.py
```

This script uses the gbifIDs from the csv files in ./data/PAI_Diptera_family_GBIF we sampled from GBIF:

```
./data/PAI_Diptera_family_GBIF/
├── family1.csv
├── family2.csv
├── ...
```

Make sure `config.yaml` points to the correct data paths.

### 3. Train the CNNs

To train the classification models, run:

```bash
python ./scene_class/scene_class_order.py
```

Training architecture and parameters are configured in `config.yaml`.

### 4. Get Classification Uncertainties

To perform inference and obtain uncertainty estimates, use:

```bash
python ./scene_class/run_predict.py
```


