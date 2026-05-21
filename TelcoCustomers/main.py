import uvicorn
import joblib
from pydantic import BaseModel
from fastapi import FastAPI
from pathlib import Path


BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / 'model_rf.pkl')
scaler = joblib.load(BASE_DIR / 'scaler.pkl')


app = FastAPI()


col = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn',
       'Contract_One year',
       'Contract_Two year',
       'InternetService_Fiber optic',
       'InternetService_No',
       'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes',
       'TechSupport_No internet service',
       'TechSupport_Yes']

class TelecomSchema(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    OnlineSecurity: str
    TechSupport: str

    class Config:
        populate_by_name = True

@app.post('/predict')
async def predict(data:TelecomSchema):
    data_dict = dict(data)

    new_contract = data_dict.pop('Contract')
    contract_binary = [
        1 if new_contract == "One year" else 0,
        1 if new_contract == "Two year" else 0]

    new_internet_service = data_dict.pop('InternetService')
    internet_binary = [
        1 if new_internet_service == "Fiber optic" else 0,
        1 if new_internet_service == "No" else 0]

    new_online_security = data_dict.pop('OnlineSecurity')
    security_binary = [
        1 if new_online_security == "No internet service" else 0,
        1 if new_online_security == "Yes" else 0]

    new_tech_support = data_dict.pop('TechSupport')
    support_binary = [
        1 if new_tech_support == "No internet service" else 0,
        1 if new_tech_support == "Yes" else 0]

    features = list(data_dict.values()) + contract_binary + internet_binary + security_binary + support_binary
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]
    return {'Churn answer': 'Yes' if prediction == 1 else 'No'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
