from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

# ----------------------------
# Snapshot FFN Model Definition
# ----------------------------
class SnapshotSepsis(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

device = torch.device("cpu")

model = SnapshotSepsis()

model.load_state_dict(
    torch.load("simple_snapshot_sepsis.pt", map_location=device)
)

model.eval()


# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sepsis snapshot model API running"}

class VitalsInput(BaseModel):
    hr: float
    rr: float
    sbp: float
    temp: float
    spo2: float
    lactate: float
    urine_ml: float

@app.post("/predict")
def predict(v: VitalsInput):

    features = np.array([
        [
            v.hr, v.rr, v.sbp, v.temp, 
            v.spo2, v.lactate, v.urine_ml
        ]
    ], dtype=np.float32)

    input_tensor = torch.tensor(features)

    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    prediction = int(probability > 0.5)

    return {
        "sepsis_detected": prediction,
        "risk_probability": probability
    }
