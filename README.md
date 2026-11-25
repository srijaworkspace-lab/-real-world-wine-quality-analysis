# Real-World Dataset Analysis: UCI Wine Quality

**Week 2 internship project** using Python, Pandas, NumPy and Matplotlib.

## Dataset
This project uses the red vinho verde wine dataset from the UCI Machine Learning Repository. It contains physicochemical measurements (such as acidity, sugar, chlorides, density, pH, sulphates and alcohol) and a sensory quality score.

Official source: https://archive.ics.uci.edu/dataset/186/wine+quality  
Citation: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. DOI: 10.24432/C56S3T.  
License: CC BY 4.0.

The CSV is downloaded automatically on the first run and saved in `data/`.

## Project features
- Import and inspect a real-world CSV dataset.
- Review data types, dataset dimensions and descriptive statistics.
- Identify missing values and duplicate rows.
- Remove duplicate records and fill missing numeric values with column medians if any exist.
- Filter high-quality samples and sort records.
- Group by quality score and calculate aggregate statistics.
- Explore feature relationships with correlation.
- Create four Matplotlib visualizations.
- Export cleaned data, grouped summaries, correlations and a text report.

## Folder structure
```text
real-world-wine-quality-analysis/
├── analysis.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── charts/
└── outputs/
```

## Run locally (Mac / VS Code)
1. Open this folder in VS Code.
2. In Terminal, install dependencies:
   `python3 -m pip install pandas numpy matplotlib`
3. Run:
   `python3 analysis.py`
4. The first run requires an internet connection to download the dataset from UCI.
5. Review the generated `charts/` and `outputs/` folders.

## Google Colab
Upload `analysis.py` to Colab or copy its code into a notebook cell. Install dependencies if needed, then run the script. An internet connection is required for the initial dataset download.

## Important interpretation note
The original UCI Wine Quality data is documented as having no missing values. The program still checks for missing values and includes a median-imputation path for robustness. It also checks and removes duplicate rows. Report the actual counts printed by your run; do not claim missing values were found if the output says zero.

Correlation indicates association, not causation.

## Outputs
- `charts/quality_distribution.png`
- `charts/alcohol_by_quality.png`
- `charts/acidity_vs_quality.png`
- `charts/correlation_matrix.png`
- `outputs/cleaned_wine_data.csv`
- `outputs/quality_group_summary.csv`
- `outputs/quality_correlations.csv`
- `outputs/analysis_summary.txt`
