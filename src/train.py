import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data import load_clean_data, ROOT

ARTIFACTS = ROOT / "artifacts"
TRACKING_URI = "sqlite:///mlflow.db"
TARGET = "target"
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("heart-disease-classification")

    candidates = [
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced"),
         {"model__C": [0.1, 1.0, 10.0]}),
        ("random_forest", RandomForestClassifier(random_state=42, class_weight="balanced"),
         {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10]})
    ]
    best_auc, best_pipeline, best_name, best_metrics = -1, None, None, None
    for name, estimator, grid in candidates:
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            model = search.best_estimator_
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "cv_best_roc_auc": float(search.best_score_)
            }
            mlflow.log_params(search.best_params_)
            mlflow.log_param("model_name", name)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")
            report = classification_report(y_test, pred, output_dict=True, zero_division=0)
            report_path = ARTIFACTS / f"{name}_classification_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            mlflow.log_artifact(str(report_path))
            print(name, metrics, "best_params=", search.best_params_)
            if metrics["roc_auc"] > best_auc:
                best_auc, best_pipeline, best_name, best_metrics = metrics["roc_auc"], model, name, metrics

    joblib.dump(best_pipeline, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "feature_names.json").write_text(json.dumps(list(X.columns), indent=2))
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "selected_model": best_name, "metrics": best_metrics
    }, indent=2))
    print(f"Saved model: {ARTIFACTS / 'model.joblib'}")
    print(f"Saved metadata: {ARTIFACTS / 'feature_names.json'}")
    print(f"Saved metrics: {ARTIFACTS / 'metrics.json'}")

if __name__ == "__main__":
    main()
