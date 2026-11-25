"""
Week 2: Real-World Dataset Analysis
Dataset: UCI Wine Quality (red vinho verde)
Run: python3 analysis.py

The script downloads the public CSV the first time it runs, then analyzes it.
"""
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "charts"
OUTPUT_DIR = ROOT / "outputs"
DATA_FILE = DATA_DIR / "winequality-red.csv"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

def load_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if not DATA_FILE.exists():
        print("Downloading the UCI Wine Quality dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    # The UCI file is semicolon-separated, not comma-separated.
    return pd.read_csv(DATA_FILE, sep=";")

def main():
    CHART_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Import and explore
    df = load_dataset()
    print("\n=== DATASET OVERVIEW ===")
    print("First five rows:\n", df.head())
    print("\nShape (rows, columns):", df.shape)
    print("\nColumn data types:\n", df.dtypes)
    print("\nDescriptive statistics:\n", df.describe().round(2))

    # 2. Missing values and duplicates
    print("\n=== DATA QUALITY CHECK ===")
    missing_before = df.isnull().sum()
    print("Missing values by column:\n", missing_before)
    duplicate_count = int(df.duplicated().sum())
    print("Duplicate rows found:", duplicate_count)

    # Remove duplicate records.
    df = df.drop_duplicates().copy()

    # If missing numeric values exist, replace them with their column median.
    numeric_columns = df.select_dtypes(include=np.number).columns
    missing_after_duplicates = int(df[numeric_columns].isnull().sum().sum())
    if missing_after_duplicates:
        df[numeric_columns] = df[numeric_columns].fillna(
            df[numeric_columns].median()
        )
        print("Missing numeric values filled using column medians.")
    else:
        print("No missing numeric values found; no imputation was necessary.")

    # 3. Filter, sort, group and aggregate
    print("\n=== FILTER / SORT / GROUP / AGGREGATE ===")
    high_quality = df[df["quality"] >= 7]
    print("High-quality records (quality >= 7):", len(high_quality))

    sorted_wines = df.sort_values(
        by=["quality", "alcohol"], ascending=[False, False]
    )
    print("\nTop 10 wines by quality, then alcohol:\n",
          sorted_wines[["quality", "alcohol", "volatile acidity",
                        "citric acid", "sulphates"]].head(10).round(3))

    grouped = df.groupby("quality").agg(
        wine_count=("quality", "size"),
        mean_alcohol=("alcohol", "mean"),
        mean_volatile_acidity=("volatile acidity", "mean"),
        mean_sulphates=("sulphates", "mean"),
        mean_density=("density", "mean")
    ).round(3)
    print("\nAggregated statistics by quality:\n", grouped)

    # 4. Relationships and correlation
    corr = df.corr(numeric_only=True)
    quality_corr = corr["quality"].drop("quality").sort_values(
        ascending=False
    )
    print("\nCorrelation of numeric features with quality:\n",
          quality_corr.round(3))

    # 5. Save cleaned and summarized data
    df.to_csv(OUTPUT_DIR / "cleaned_wine_data.csv", index=False)
    grouped.to_csv(OUTPUT_DIR / "quality_group_summary.csv")
    quality_corr.to_csv(OUTPUT_DIR / "quality_correlations.csv",
                        header=["correlation"])

    # 6. Visualizations
    plt.figure(figsize=(8, 5))
    df["quality"].value_counts().sort_index().plot(kind="bar")
    plt.title("Distribution of Wine Quality Scores")
    plt.xlabel("Quality score")
    plt.ylabel("Number of wine samples")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "quality_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    df.groupby("quality")["alcohol"].mean().plot(kind="bar")
    plt.title("Average Alcohol Content by Quality Score")
    plt.xlabel("Quality score")
    plt.ylabel("Average alcohol")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "alcohol_by_quality.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["volatile acidity"], df["quality"], alpha=0.35)
    plt.title("Volatile Acidity vs Wine Quality")
    plt.xlabel("Volatile acidity")
    plt.ylabel("Quality score")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "acidity_vs_quality.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns,
               rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Correlation Matrix of Wine Features")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlation_matrix.png", dpi=150)
    plt.close()

    # 7. Short evidence-based summary
    strongest_positive = quality_corr.idxmax()
    strongest_negative = quality_corr.idxmin()
    summary = [
        "WEEK 2: REAL-WORLD DATA ANALYSIS SUMMARY",
        "Dataset: UCI Wine Quality - red vinho verde samples",
        f"Rows after duplicate removal: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Duplicate rows removed: {duplicate_count}",
        f"Missing values before cleaning: {int(missing_before.sum())}",
        f"Records with quality >= 7: {len(high_quality)}",
        f"Mean quality score: {df['quality'].mean():.3f}",
        f"Mean alcohol: {df['alcohol'].mean():.3f}",
        f"Strongest positive feature correlation with quality: "
        f"{strongest_positive} ({quality_corr[strongest_positive]:.3f})",
        f"Strongest negative feature correlation with quality: "
        f"{strongest_negative} ({quality_corr[strongest_negative]:.3f})",
        "",
        "Interpretation: Correlation describes association, not causation.",
        "Wine quality scores are sensory ratings; this analysis does not prove",
        "that changing one chemical measurement will cause a quality change."
    ]
    summary_path = OUTPUT_DIR / "analysis_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print("\n".join(summary))
    print("\nDone. See the charts/ and outputs/ folders.")

if __name__ == "__main__":
    main()
