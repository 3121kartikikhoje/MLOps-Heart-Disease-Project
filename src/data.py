from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SOURCE_FILE = RAW_DIR / "processed.cleveland.data"
CSV_FILE = RAW_DIR / "heart_disease.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def acquire_data() -> Path:
    """Convert the uploaded UCI Cleveland source file into a named CSV."""
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset source is missing: {SOURCE_FILE}. "
            "Keep data/raw/processed.cleveland.data in the project."
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE_FILE, header=None, names=COLUMNS, na_values="?")
    df.to_csv(CSV_FILE, index=False)
    return CSV_FILE

def load_clean_data(download_if_missing: bool = True) -> pd.DataFrame:
    """Load the Cleveland subset and convert disease severity 0-4 to binary."""
    if not CSV_FILE.exists():
        if not SOURCE_FILE.exists():
            raise FileNotFoundError(
                f"Dataset not found. Expected {SOURCE_FILE}; see README.md."
            )
        acquire_data()
    df = pd.read_csv(CSV_FILE)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = (df["target"] > 0).astype(int)
    return df

def save_clean_data() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()
    path = PROCESSED_DIR / "heart_disease_clean.csv"
    df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    print(f"CSV created: {acquire_data()}")
    print(f"Cleaned data saved: {save_clean_data()}")
