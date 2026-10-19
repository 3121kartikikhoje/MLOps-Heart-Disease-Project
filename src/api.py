from pathlib import Path
import logging
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-api")
REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
REQUEST_LATENCY = Histogram("prediction_request_duration_seconds", "Prediction request duration")

FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
app.mount("/metrics", make_asgi_app())

class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    logger.info("%s %s status=%s duration_ms=%.2f", request.method, request.url.path,
                response.status_code, (time.time()-started)*1000)
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "docs": "/docs", "predict": "/predict"}

@app.post("/predict")
def predict(patient: PatientFeatures):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact missing. Run python -m src.train first.")
    try:
        model = joblib.load(MODEL_PATH)
        row = pd.DataFrame([patient.model_dump()], columns=FEATURES)
        with REQUEST_LATENCY.time():
            pred = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0][1])
        REQUEST_COUNT.inc()
        return {"prediction": pred, "label": "higher_risk_class" if pred == 1 else "lower_risk_class",
                "confidence": probability, "disclaimer": "Educational demonstration only; not medical advice."}
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed; check API logs.") from exc
