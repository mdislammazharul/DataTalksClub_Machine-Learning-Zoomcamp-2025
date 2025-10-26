import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the pipeline model
with open('pipeline_v1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

# Define input schema
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Create FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(client: Client):
    data = client.dict()
    proba = model.predict_proba([data])[0, 1]
    return {"subscription_probability": round(float(proba), 3)}