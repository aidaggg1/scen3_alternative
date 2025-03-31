from fastapi import FastAPI
from pydantic import BaseModel, validator

class CarMaintenanceCostCalculator:
    def __init__(self, brand, model, year, fuel_type, transmission, mileage, engine_CC):
        self.brand = brand
        self.model = model
        self.year = year
        self.fuel_type = fuel_type
        self.transmission = transmission
        self.mileage = mileage
        self.engine_CC = engine_CC

    #based on the fuel type the car maintenance will be different
    def get_fuel_type_factor(self):
        fuel_type_factors = {
            "Petrol": 45,
            "Diesel": 70,
            "CNG": 30,
            "Electric": 15
        }
        return fuel_type_factors.get(self.fuel_type, 0)

    #same with transmission
    def get_transmission_factor(self):
        transmission_factors = {
            "Manual": 35,
            "Automatic": 60
        }
        return transmission_factors.get(self.transmission, 0)

    #formula parameters
    def calculate_cost(self):
        base_cost = 100
        cost_per_mile = 0.05
        cost_per_cc = 0.02
        year_factor = (2025 - self.year) * 10
        fuel_type_factor = self.get_fuel_type_factor()
        transmission_factor = self.get_transmission_factor()  #

        #calculate maintenance cost
        maintenance_cost = (base_cost +
                            (self.mileage * cost_per_mile) +
                            (self.engine_CC * cost_per_cc) +
                            year_factor +
                            fuel_type_factor +
                            transmission_factor)

        return maintenance_cost

#api

VALID_BRANDS = ['Honda', 'Hyundai', 'Kia', 'Mahindra', 'Maruti Suzuki', 'Renault', 'Skoda', 'Tata Motors', 'Toyota', 'Volkswagen']
VALID_MODELS = ['Altroz', 'Amaze', 'Baleno', 'Bolero', 'Camry', 'Carens', 'Carnival', 'City', 'Civic', 'Creta', 'Duster', 'Dzire',
                'EV6', 'Ertiga', 'Fortuner', 'Glanza', 'Harrier', 'Innova', 'Jazz', 'Kiger', 'Kushaq', 'Kwid', 'Lodgy', 'Nexon', 'Octavia',
                'Polo', 'Punch', 'Rapid', 'Scorpio', 'Seltos', 'Slavia', 'Sonet', 'Superb', 'Swift', 'Taigun', 'Thar', 'Tiago', 'Tiguan',
                'Triber', 'Urban Cruiser', 'Vento', 'Venue', 'Verna', 'Virtus', 'WR-V', 'WagonR', 'XUV300', 'XUV700', 'i10', 'i20']
VALID_FUEL_TYPES = ['CNG', 'Diesel', 'Electric', 'Petrol']
VALID_TRANSMISSIONS = ['Manual', 'Automatic']

app = FastAPI()

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

#client can choose which model use when executing the api
@app.post("/predict")
async def predict(car_data: CarData):

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

    return {"Service cost using formula is ": formula}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)