from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import joblib
import json
import numpy as np
import tensorflow as tf
from formula import CarMaintenanceCostCalculator

app = FastAPI()

#load models
model_simple_dnn = tf.keras.models.load_model('models/car_cost_model1.h5')
model_dnn = tf.keras.models.load_model('models/car_cost_model2.h5')
model_random_forest = joblib.load('models/car_cost_random_forest.pkl')
    # --> the most accurate is the 2, then 1 and the least the 3rd !!

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
scaler = joblib.load('encoders/scaler_cars.pkl')

class CarData(BaseModel):
    Brand: str
    Model: str
    Year: int
    Fuel_Type: str
    Transmission: str
    Mileage: float
    Engine_CC: float

#transform the brand, model, fuel type, transmission to its associated number
def preprocess_input(data: CarData):
    data_dict = data.dict()
    data_dict['Brand'] = brand_to_code[data_dict['Brand']]
    data_dict['Model'] = model_to_code[data_dict['Model']]
    data_dict['Fuel_Type'] = fuel_to_code[data_dict['Fuel_Type']]
    data_dict['Transmission'] = 0 if data_dict['Transmission'] == 'Manual' else 1
    return np.array([[data_dict['Brand'], data_dict['Model'], data_dict['Year'], data_dict['Fuel_Type'], data_dict['Transmission'], data_dict['Mileage'], data_dict['Engine_CC']]])

#client can choose which model use when executing the api
@app.post("/predict")
async def predict(car_data: CarData):
    input_data = preprocess_input(car_data)
    input_data = scaler.transform(input_data)

    #calculate the cost using all models
    prediction_simple_dnn = model_simple_dnn.predict(input_data)
    prediction_dnn = model_dnn.predict(input_data)
    prediction_random_forest = model_random_forest.predict(input_data)

    #using formula
    calculator = CarMaintenanceCostCalculator(
        brand=car_data.Brand,
        model=car_data.Model,
        year=car_data.Year,
        fuel_type=car_data.Fuel_Type,
        transmission=car_data.Transmission,
        mileage=car_data.Mileage,
        engine_CC=car_data.Engine_CC
    )
    formula = calculator.calculate_cost()

    return {
        "Service cost using simple DNN is": float(prediction_simple_dnn[0]),
        "Service cost using DNN is ": float(prediction_dnn[0]),
        "Service cost using random forest is ": float(prediction_random_forest[0]),
        "Service cost using formula is ": formula
    }
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)