import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import load_clean_data, ROOT

def run_eda():
    df = load_clean_data()
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df.hist(figsize=(14, 10), bins=15)
    plt.tight_layout()
    plt.savefig(out / "feature_histograms.png", dpi=160)
    plt.close()
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png", dpi=160)
    plt.close()
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Heart disease target class balance")
    plt.tight_layout()
    plt.savefig(out / "class_balance.png", dpi=160)
    plt.close()
    print("Rows/columns:", df.shape)
    print("Missing values:\n", df.isna().sum())
    print("Target balance:\n", df["target"].value_counts(normalize=True))
    print(f"Figures saved to {out}")

if __name__ == "__main__":
    run_eda()
