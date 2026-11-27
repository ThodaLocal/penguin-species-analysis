# Penguin Species Analysis

**CSE 101 Capstone Project — Team T08**
Programming for Problem Solving Using Python · K.R. Mangalam University

An analysis of the Palmer Penguins dataset (built into Seaborn) that compares the Adelie, Chinstrap and Gentoo penguin species by their body measurements, visualises the patterns, and predicts a penguin's species from its measurements using a nearest-centroid method written in plain Python with Pandas.

## Team

| Member | Roll No. |
|---|---|
| Mehul Rao | 2601730096 |
| Jay Talan | 2601730196 |
| Vivek | 2601730115 |
| Vedant Bhardwaj | 2601730117 |
| Aryan | 2601730141 |

Supervisor: Dr. Ravinder Beniwal

## Key Results

- Dataset: 344 penguins → 333 after removing incomplete records (19 missing cells)
- Species: Adelie 146 · Gentoo 119 · Chinstrap 68
- Gentoo is the heaviest species (avg 5092.44 g); Chinstrap has the longest bills (avg 48.83 mm)
- Chinstrap penguins appear only on Dream island and Gentoo only on Biscoe island in this dataset
- Species prediction accuracy: **97.01%** (65 of 67 unseen records)

## How to Run

```bash
pip install pandas seaborn matplotlib
python penguin_analysis.py
python test_penguin_analysis.py
```

`penguin_analysis.py` prints a summary and writes charts to `charts/` and results (`species_averages.csv`, `stats.json`) to `results/`. `test_penguin_analysis.py` runs 10 automated checks — all pass.

## Files

- `penguin_analysis.py` — main analysis program
- `test_penguin_analysis.py` — automated checks
- `Project_Report_Penguin_Species_Analysis_T08.docx` — full project report
- `charts/` — six generated charts
- `results/` — generated statistics files
