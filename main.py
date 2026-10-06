
from fastapi import FastAPI
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="Amazon Review ML API")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.post("/predict")
def predict(data: dict):

    review = data["review"]

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    return {
        "prediction": int(prediction),
        "class_name": "Positive" if prediction == 1 else "Negative"
    }
    
