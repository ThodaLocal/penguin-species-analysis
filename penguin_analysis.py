import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(BASE, "charts")
RESULTS = os.path.join(BASE, "results")
os.makedirs(CHARTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

sns.set_theme(style="whitegrid")


def load_data():
    df = sns.load_dataset("penguins")
    return df


def clean_data(df):
    missing_before = int(df.isna().sum().sum())
    clean = df.dropna().reset_index(drop=True)
    return clean, missing_before


def nearest_centroid_accuracy(df):
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(shuffled) * 0.8)
    train, test = shuffled.iloc[:split], shuffled.iloc[split:]
    center = train[FEATURES].mean()
    scale = train[FEATURES].std()
    train_z = train.copy()
    test_z = test.copy()
    train_z[FEATURES] = (train[FEATURES] - center) / scale
    test_z[FEATURES] = (test[FEATURES] - center) / scale
    centroids = train_z.groupby("species")[FEATURES].mean()
    correct = 0
    predictions = []
    for _, row in test_z.iterrows():
        distances = ((centroids - row[FEATURES]) ** 2).sum(axis=1)
        predicted = distances.idxmin()
        predictions.append(predicted)
        if predicted == row["species"]:
            correct += 1
    accuracy = round(correct / len(test) * 100, 2)
    return accuracy, len(test), correct


def make_charts(df):
    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x="species", hue="species", palette="deep", legend=False)
    plt.title("Number of Penguins per Species")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "fig1_species_count.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", s=55)
    plt.title("Bill Length vs Bill Depth by Species")
    plt.xlabel("Bill Length (mm)")
    plt.ylabel("Bill Depth (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "fig2_bill_scatter.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="species", y="body_mass_g", hue="species", palette="deep", legend=False)
    plt.title("Body Mass Distribution by Species")
    plt.xlabel("Species")
    plt.ylabel("Body Mass (g)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "fig3_body_mass_box.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.histplot(data=df, x="flipper_length_mm", hue="species", bins=20, kde=True, alpha=0.55)
    plt.title("Flipper Length Distribution by Species")
    plt.xlabel("Flipper Length (mm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "fig4_flipper_hist.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x="island", hue="species", palette="deep")
    plt.title("Species Found on Each Island")
    plt.xlabel("Island")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "fig5_island_species.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 5))
    corr = df[FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Correlation Between Body Measurements")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "fig6_correlation_heatmap.png"), dpi=150)
    plt.close()


def main():
    df = load_data()
    raw_rows = len(df)
    clean, missing_cells = clean_data(df)
    clean_rows = len(clean)

    species_counts = clean["species"].value_counts().to_dict()
    island_counts = clean["island"].value_counts().to_dict()
    sex_counts = clean["sex"].value_counts().to_dict()
    species_island = pd.crosstab(clean["island"], clean["species"]).to_dict()

    means = clean.groupby("species")[FEATURES].mean().round(2)
    means.to_csv(os.path.join(RESULTS, "species_averages.csv"))

    accuracy, test_size, correct = nearest_centroid_accuracy(clean)
    make_charts(clean)

    stats = {
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "rows_removed": raw_rows - clean_rows,
        "missing_cells": missing_cells,
        "columns": list(df.columns),
        "species_counts": {k: int(v) for k, v in species_counts.items()},
        "island_counts": {k: int(v) for k, v in island_counts.items()},
        "sex_counts": {k: int(v) for k, v in sex_counts.items()},
        "species_island": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in species_island.items()},
        "species_means": {sp: {f: float(means.loc[sp, f]) for f in FEATURES} for sp in means.index},
        "overall_means": {f: round(float(clean[f].mean()), 2) for f in FEATURES},
        "classifier": {"test_size": test_size, "correct": correct, "accuracy_percent": accuracy},
    }
    with open(os.path.join(RESULTS, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    print("=== Penguin Species Analysis ===")
    print(f"Raw dataset: {raw_rows} rows x {df.shape[1]} columns")
    print(f"Missing cells found: {missing_cells}")
    print(f"Clean dataset: {clean_rows} rows (removed {raw_rows - clean_rows} incomplete rows)")
    print("\nPenguins per species (clean data):")
    for sp, n in species_counts.items():
        print(f"  {sp}: {n}")
    print("\nAverage measurements by species:")
    print(means.to_string())
    print(f"\nNearest-centroid species prediction: {correct}/{test_size} correct = {accuracy}% accuracy")
    print("\nCharts saved to charts/ and results saved to results/")


if __name__ == "__main__":
    main()
