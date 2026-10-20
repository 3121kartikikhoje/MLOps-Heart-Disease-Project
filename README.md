# MLOps Assignment 01 — Heart Disease Prediction

Course: MLOps AIMLCZG523 | Assignment total: 50 marks

This project uses the UCI Heart Disease Cleveland processed dataset supplied in `data/raw/processed.cleveland.data`. It demonstrates data preparation, EDA, model comparison, MLflow tracking, reproducibility, automated tests, GitHub Actions, FastAPI, Docker, Kubernetes manifests and Prometheus-compatible metrics.

> Educational demonstration only. Not a medical device; do not use for diagnosis or treatment.

## A. Exact Windows folder path

The intended project root is:

`C:\Users\KARTIKI KHOJE\Documents\MLOPS ASSIGNMENT\MLOps_Heart_Disease_Project\MLOps_Heart_Disease_Project`

Extract this ZIP into:

`C:\Users\KARTIKI KHOJE\Documents\MLOPS ASSIGNMENT\MLOps_Heart_Disease_Project`

The extracted project folder should then be the nested path above. Open that inner folder in VS Code: it must contain `README.md`, `requirements.txt`, `src`, `tests`, `data`, and `Dockerfile`.

In PowerShell, go to the root using:
```powershell
Set-Location "C:\Users\KARTIKI KHOJE\Documents\MLOPS ASSIGNMENT\MLOps_Heart_Disease_Project\MLOps_Heart_Disease_Project"
Get-Location
Get-ChildItem
```

## B. Environment setup (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. If `py -3.11` is not found, install Python 3.11 and reopen VS Code.

## C. Dataset, EDA and training

The source dataset is included. No dataset download is required.

```powershell
python -m src.data
python -m src.eda
python -m src.train
```

The Cleveland dataset has 14 columns and target values 0-4. The target is converted to binary: 0 = no disease; 1 = any target value greater than 0. The source uses `?` for missing feature values; the preprocessing pipeline imputes them.

Verify artifacts:
```powershell
Test-Path .\artifacts\model.joblib
Test-Path .\artifacts\feature_names.json
Test-Path .\artifacts\metrics.json
```
All must return `True` after training succeeds.

## D. MLflow tracking

From the project root, in a second terminal:
```powershell
.\.venv\Scripts\Activate.ps1
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000. Run `python -m src.train` from another terminal at the same project root. The UI and training use `sqlite:///mlflow.db`.

## E. Tests and lint

```powershell
pytest -q
ruff check src tests
```

Fix any reported failures before continuing.

## F. Run API locally

```powershell
uvicorn src.api:app --reload
```
Open:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/metrics

Use Swagger UI to send a POST request to `/predict`:
```json
{
  "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
  "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
}
```
This is only a technical API smoke-test example, not a clinical recommendation.

## G. Docker

Train first, then:
```powershell
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```
Check `/health`, `/docs`, `/predict`, `/metrics`. Docker Desktop must be running.

## H. Local Kubernetes with Minikube

Install Minikube and kubectl first. With Docker Desktop running:
```powershell
minikube start
docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get deployments
kubectl get services
kubectl port-forward service/heart-disease-api 8001:8000
```
In another terminal, test http://127.0.0.1:8001/health and http://127.0.0.1:8001/docs. Port-forward is a local demonstration, not a public cloud URL.

## I. GitHub

Create a repository, then from this project root:
```powershell
git init
git add .
git status
git commit -m "Add end-to-end MLOps heart disease project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
Replace the placeholder with your actual repository URL. In GitHub's Actions tab, inspect the `MLOps CI` run and save evidence of the lint, tests, EDA, training and uploaded artifacts. Workflow failures must be fixed, not hidden.

## J. Evidence and final submission

Save genuine screenshots under `reports/screenshots/`:
1. Dataset shape/missing values and EDA figures.
2. MLflow experiment runs, parameters, metrics and artifacts.
3. Successful Pytest and Ruff output.
4. GitHub Actions successful workflow and artifact upload.
5. Docker build/run and API docs/prediction.
6. Kubernetes pod/deployment/service and API verification.
7. Monitoring metrics/logs and dashboard if configured.
8. `reports/REPORT_TEMPLATE.md` into a 10-page DOCX or PDF. Include actual results, figures with interpretations, architecture diagram, setup steps, repository URL, deployment URL or local access instructions, and a short walkthrough video. Do not fabricate results or screenshots.
