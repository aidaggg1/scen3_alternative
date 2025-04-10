from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel, validator
import joblib
import json
import numpy as np

VALID_BRANDS = ['Honda', 'Hyundai', 'Kia', 'Mahindra', 'Maruti Suzuki', 'Renault', 'Skoda', 'Tata Motors', 'Toyota', 'Volkswagen']
VALID_MODELS = ['Altroz', 'Amaze', 'Baleno', 'Bolero', 'Camry', 'Carens', 'Carnival', 'City', 'Civic', 'Creta', 'Duster', 'Dzire',
                'EV6', 'Ertiga', 'Fortuner', 'Glanza', 'Harrier', 'Innova', 'Jazz', 'Kiger', 'Kushaq', 'Kwid', 'Lodgy', 'Nexon', 'Octavia',
                'Polo', 'Punch', 'Rapid', 'Scorpio', 'Seltos', 'Slavia', 'Sonet', 'Superb', 'Swift', 'Taigun', 'Thar', 'Tiago', 'Tiguan',
                'Triber', 'Urban Cruiser', 'Vento', 'Venue', 'Verna', 'Virtus', 'WR-V', 'WagonR', 'XUV300', 'XUV700', 'i10', 'i20']
VALID_FUEL_TYPES = ['CNG', 'Diesel', 'Electric', 'Petrol']
VALID_TRANSMISSIONS = ['Manual', 'Automatic']

app = FastAPI()

#load model
model_random_forest = joblib.load('models/car_cost_random_forest.pkl')

#load encoders
with open('encoders/brand_codes.json', 'r') as f:
    brand_codes = json.load(f)
with open('encoders/model_codes.json', 'r') as f:
    model_codes = json.load(f)
with open('encoders/fuel_type_codes.json', 'r') as f:
    fuel_type_codes = json.load(f)

brand_to_code = {v: k for k, v in brand_codes.items()}
model_to_code = {v: k for k, v in model_codes.items()}
fuel_to_code = {v: k for k, v in fuel_type_codes.items()}

#load the scaler
scaler = joblib.load('scalers/scaler_cars.pkl')

class CarData(BaseModel):
    Brand: str
    Model: str
    Year: int
    Fuel_Type: str
    Transmission: str
    Mileage: float
    Engine_CC: float

    @validator("Brand")
    def validate_brand(cls, v):
        if v not in VALID_BRANDS:
            raise ValueError(f"Invalid brand, it must be one of: {', '.join(VALID_BRANDS)}")
        return v

    @validator("Model")
    def validate_model(cls, v):
        if v not in VALID_MODELS:
            raise ValueError(f"Invalid model, it must be one of: {', '.join(VALID_MODELS)}")
        return v

    @validator("Fuel_Type")
    def validate_fuel_type(cls, v):
        if v not in VALID_FUEL_TYPES:
            raise ValueError(f"Invalid fuel type, it must be one of: {', '.join(VALID_FUEL_TYPES)}")
        return v

    @validator("Transmission")
    def validate_transmission(cls, v):
        if v not in VALID_TRANSMISSIONS:
            raise ValueError(f"Invalid transmission, it must be one of: {', '.join(VALID_TRANSMISSIONS)}")
        return v

#transform the brand, model, fuel type, transmission to its associated number
def preprocess_input(data: CarData):
    data_dict = data.model_dump()
    data_dict['Brand'] = brand_to_code[data_dict['Brand']]
    data_dict['Model'] = model_to_code[data_dict['Model']]
    data_dict['Fuel_Type'] = fuel_to_code[data_dict['Fuel_Type']]
    data_dict['Transmission'] = 0 if data_dict['Transmission'] == 'Manual' else 1
    return np.array([[data_dict['Brand'], data_dict['Model'], data_dict['Year'], data_dict['Fuel_Type'], data_dict['Transmission'], data_dict['Mileage'], data_dict['Engine_CC']]])

@app.post("/predict")
async def predict(car_data: CarData):
    input_data = preprocess_input(car_data)
    input_data = scaler.transform(input_data)

    #make the prediction
    prediction_random_forest = model_random_forest.predict(input_data)
    response = round(float(prediction_random_forest[0]), 2)

    return {"Service cost using random forest is": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)