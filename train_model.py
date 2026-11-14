
from Data_Loader import load_data
from ML_model import train_models

def main():

    print("Loading CO2 dataset...")

    df = load_data()

    print("Dataset shape:", df.shape)

    print("\nTraining machine learning models...")

    best_model, results = train_models(df)

    print("\nTraining completed successfully!")

    print("\nModel Performance:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
