
import pandas as pd

# Load Excel dataset
FILE_PATH = "Data/CO2_Emission.xlsx"

def load_data():
    df = pd.read_excel(FILE_PATH)
    return df


def inspect_data(df):

    print("\n--- FIRST 5 RECORDS ---")
    print(df.head())

    print("\n--- DATASET SHAPE ---")
    print(df.shape)

    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- STATISTICAL SUMMARY ---")
    print(df.describe(include="all"))


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
