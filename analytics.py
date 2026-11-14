
import pandas as pd

from Data_Loader import load_data

# Load dataset
df = load_data()

CO2_COL = "CO2 Emissions(g/km)"


# 1. Total CO2 emissions
def total_emissions():
    return df[CO2_COL].sum()


# 2. Average CO2 emissions
def average_emissions():
    return df[CO2_COL].mean()


# 3. Maximum emissions
def maximum_emissions():
    return df[CO2_COL].max()


# 4. Minimum emissions
def minimum_emissions():
    return df[CO2_COL].min()


# 5. Emissions by vehicle class
def emissions_by_vehicle_class():
    return (
        df.groupby("Vehicle Class")[CO2_COL]
        .mean()
        .sort_values(ascending=False)
    )


# 6. Emissions by fuel type
def emissions_by_fuel_type():
    return (
        df.groupby("Fuel Type")[CO2_COL]
        .mean()
        .sort_values(ascending=False)
    )


# 7. Top 10 highest emission vehicles
def top_10_emission_vehicles():
    return df.nlargest(10, CO2_COL)

#8. emission based on cylinder
def emissions_by_cylinders():
    return (
        df.groupby("Cylinders")[CO2_COL]
        .mean()
        .sort_values(ascending=False)
    )

#9. emission based on transmission
def emissions_by_Transmission():
    return (
        df.groupby("Transmission")[CO2_COL]
        .mean()
        .sort_values(ascending=False)
    )

if __name__ == "__main__":

    print("\nTOTAL CO2 EMISSIONS:", total_emissions())

    print("\nAVERAGE CO2 EMISSIONS:", average_emissions())

    print("\nMAXIMUM EMISSIONS:", maximum_emissions())

    print("\nMINIMUM EMISSIONS:", minimum_emissions())

    print("\nEMISSIONS BY VEHICLE CLASS:")
    print(emissions_by_vehicle_class())

    print("\nEMISSIONS BY FUEL TYPE:")
    print(emissions_by_fuel_type())

    print("\nEMISSIONS BY Transmission:")
    print(emissions_by_Transmission())

    print("\nTOP 10 EMISSION VEHICLES:")
    print(top_10_emission_vehicles())

    print("\nAVERAGE CO2 EMISSIONS BY CYLINDERS:")
    print(emissions_by_cylinders())
