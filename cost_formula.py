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