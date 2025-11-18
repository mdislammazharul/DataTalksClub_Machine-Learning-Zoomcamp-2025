import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the pipeline model
with open('heart_failure_model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

# Define input schema
class PatientData(BaseModel):
    age: float = Field(..., json_schema_extra={"example": 65})
    anaemia: int = Field(..., json_schema_extra={"example": 0})
    creatinine_phosphokinase: int = Field(..., json_schema_extra={"example": 250})
    diabetes: int = Field(..., json_schema_extra={"example": 1})
    ejection_fraction: int = Field(..., json_schema_extra={"example": 35})
    high_blood_pressure: int = Field(..., json_schema_extra={"example": 1})
    platelets: float = Field(..., json_schema_extra={"example": 250000})
    serum_creatinine: float = Field(..., json_schema_extra={"example": 1.9})
    serum_sodium: int = Field(..., json_schema_extra={"example": 130})
    sex: int = Field(..., json_schema_extra={"example": 1})
    smoking: int = Field(..., json_schema_extra={"example": 0})
    time: int = Field(..., json_schema_extra={"example": 120})

# Create FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"hello": "world"}

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)[0]  # 0 or 1
        probability = model.predict_proba(input_df)[0, 1]  # probability of class 1 (death)
        
        return {
            "prediction": int(prediction),
            "probability_death": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))