from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.model_training import train_models
from src.evaluation import evaluate_models
from src.prediction import make_predictions
from src.recommendation import recommend_restaurants

def main():
    df = load_data("data/restaurents.csv")  # use your file name

    df = clean_data(df)
    df = engineer_features(df)

    models, X_test, y_test = train_models(df)

    evaluate_models(models, X_test, y_test)

    make_predictions(models, X_test)

    recs = recommend_restaurants(df)
    print("Sample Recommendations:", recs)

if __name__ == "__main__":
    main()