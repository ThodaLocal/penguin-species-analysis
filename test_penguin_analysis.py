import json
import os

import pandas as pd
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

results = []


def check(test_id, name, condition):
    results.append((test_id, name, "Pass" if condition else "Fail"))


df = sns.load_dataset("penguins")
clean = df.dropna().reset_index(drop=True)

check("TC-01", "Dataset loads with 344 rows and 7 columns", df.shape == (344, 7))
check("TC-02", "All expected columns are present",
      list(df.columns) == ["species", "island", "bill_length_mm", "bill_depth_mm",
                           "flipper_length_mm", "body_mass_g", "sex"])
check("TC-03", "Missing cells before cleaning equal 19", int(df.isna().sum().sum()) == 19)
check("TC-04", "Clean data has 333 rows and no missing values",
      len(clean) == 333 and int(clean.isna().sum().sum()) == 0)
counts = clean["species"].value_counts().to_dict()
check("TC-05", "Species counts are Adelie 146, Gentoo 119, Chinstrap 68",
      counts == {"Adelie": 146, "Gentoo": 119, "Chinstrap": 68})
means = clean.groupby("species")[FEATURES].mean()
check("TC-06", "Gentoo has the highest average body mass",
      means["body_mass_g"].idxmax() == "Gentoo")
check("TC-07", "Chinstrap penguins are found only on Dream island",
      set(clean[clean["species"] == "Chinstrap"]["island"]) == {"Dream"})
chart_files = ["fig1_species_count.png", "fig2_bill_scatter.png", "fig3_body_mass_box.png",
               "fig4_flipper_hist.png", "fig5_island_species.png", "fig6_correlation_heatmap.png"]
check("TC-08", "All 6 chart images are generated",
      all(os.path.getsize(os.path.join(BASE, "charts", f)) > 5000 for f in chart_files))
with open(os.path.join(BASE, "results", "stats.json")) as fh:
    stats = json.load(fh)
check("TC-09", "Species prediction accuracy is at least 90 percent",
      stats["classifier"]["accuracy_percent"] >= 90)
avg = pd.read_csv(os.path.join(BASE, "results", "species_averages.csv"))
check("TC-10", "Species averages file has one row per species", len(avg) == 3)

print(f"{'ID':<7} {'Test case':<58} Result")
for tid, name, status in results:
    print(f"{tid:<7} {name:<58} {status}")
passed = sum(1 for _, _, s in results if s == "Pass")
print(f"\nTotal: {len(results)} test cases, {passed} passed")
